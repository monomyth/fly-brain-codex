"""Assemble verified live sensory features and explicit targets for CUDA fitting."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from fly_brain.assets import write_json,sha256_file,artifact_manifest,canonical_hash,checked_files
from fly_brain.adapters import JOINT_LOWER,JOINT_UPPER
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.motor_training import save_checkpoint
from fly_brain.visual_dopamine.trajectory_training import load_demonstrations
from fly_brain.visual_dopamine.feedback_training import demonstration_weights
from fly_brain.kinematics import ToolKinematics,demonstration_target_poses
from probability_gripper import neural_aperture


def merge_episode_splits(episodes,cases):
    configurations={name:[] for name in ['train','validation','test']}
    configurations['episode_splits']={}
    assigned={}
    for entry in [*episodes,*cases]:
        point=entry.get('cube_xy_mm',entry.get('task',{}).get('cube_xy_mm'))
        point=tuple(point);split=entry['split']
        if point in assigned and assigned[point]!=split:
            raise ValueError('A held-out placement cannot move into another split: '+str(point))
        assigned[point]=split
        if list(point) not in configurations[split]:configurations[split].append(list(point))
        if 'episode' in entry:configurations['episode_splits'][entry['episode']]=split
    return configurations


def require_matching_sensory_contract(source,collector):
    """A different head can visit recovery states; its sensory encoding must match."""
    for key in ['graph_id','circuit_id','motor_circuit_id','feedback_circuit_id','temporal_sensory','observation_schema','architecture']:
        if source.metadata.get(key)!=collector.metadata.get(key):raise ValueError('Collection sensory contract mismatch: '+key)
    for key in ['retinal_mean','retinal_std','kc_mean','kc_std']:
        if not np.array_equal(getattr(source.core,key),getattr(collector.core,key)):
            raise ValueError('Collection calibration mismatch: '+key)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());root=Path(plan['project']);report=a.plan.resolve().parent
    source=Path(plan['warm_start']);dataset=Path(plan['dataset'])
    if a.output.exists() or not a.output.resolve().is_relative_to(root/'data/checkpoints'):
        raise ValueError('Use a new project-local input checkpoint')
    torch.set_num_threads(2)
    collection=json.loads((dataset/'collection-report.json').read_text())
    if {r['case_id'] for r in collection['episodes']}!={c['id'] for c in plan['collection_cases']}:
        raise ValueError('Finish all declared collection cases before fitting')
    policy=MotorPolicy(source,root/'data');meta=policy.metadata
    collection_source=Path(plan.get('collection_checkpoint',source))
    collector=policy if collection_source.resolve()==source.resolve() else MotorPolicy(collection_source,root/'data')
    require_matching_sensory_contract(policy,collector)
    feature_source=Path(plan.get('feature_checkpoint',source))
    feature_meta=json.loads((feature_source/'manifest.json').read_text())
    checked_files(feature_source,feature_meta)
    if sha256_file(feature_source/'model.npz')!=sha256_file(source/'model.npz'):
        raise ValueError('Additional feature source must preserve the identical actor and calibration')
    for key in ['graph_id','circuit_id','motor_circuit_id','feedback_circuit_id','temporal_sensory','observation_schema','architecture']:
        if feature_meta.get(key)!=meta.get(key):raise ValueError('Feature source contract mismatch: '+key)
    if collection['checkpoint_hashes']['model_sha256']!=sha256_file(collection_source/'model.npz') or collection['checkpoint_hashes']['manifest_sha256']!=sha256_file(collection_source/'manifest.json'):
        raise ValueError('Collected features use a different checkpoint or calibration')
    configs=merge_episode_splits(feature_meta['episodes'],plan['collection_cases'])
    case_splits={c['id']:c['split'] for c in plan['collection_cases']}
    for item in collection['episodes']:
        if item.get('episode_directory'):
            configs['episode_splits'][Path(item['episode_directory']).name]=case_splits[item['case_id']]
    write_json(report/'training-configurations.json',configs)
    datasets=[*feature_meta['datasets'],str(dataset.resolve())]
    examples,episodes=load_demonstrations(datasets,configs,goal_supervision=meta['goal_supervision'],sample_hz=meta['sample_hz'],
                                         grasp_goal_height_mm=meta['grasp_goal_height_mm'],recovery_target_policy=meta['recovery_target_policy'])
    new_names={Path(r['episode_directory']).name for r in collection['episodes'] if r.get('episode_directory')}
    admitted=[e for e in episodes if e['episode'] in new_names]
    counts={split:sum(e['split']==split for e in admitted) for split in ['train','validation','test']}
    if counts['train']<4 or counts['validation']<1 or counts['test']<1:
        raise ValueError(f'Insufficient verified recovery coverage: {counts}')
    old_rows=json.loads((feature_source/'feature-rows.json').read_text())
    with np.load(feature_source/'training-features.npz',allow_pickle=False) as saved:old_features=saved['features'].copy()
    old_lookup={(r['episode'],r['step']):f for r,f in zip(old_rows,old_features)}
    live={};source_records=[]
    for episode in admitted:
        folder=dataset/episode['episode'];manifest=json.loads((folder/'manifest.json').read_text())
        if manifest.get('live_feature_contract')!='normalized-encoder-output-v1' or manifest.get('policy_model_sha256')!=sha256_file(collection_source/'model.npz'):
            raise ValueError('Unrecognized live feature origin')
        if sha256_file(folder/'live-features.npz')!=manifest['live_features_sha256']:
            raise ValueError('Live feature checksum mismatch')
        rows=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
        with np.load(folder/'live-features.npz',allow_pickle=False) as saved:
            features=saved['features'].copy();frames=saved['frame_ids'].copy()
        if features.shape!=(len(rows),len(policy.circuit.kc)) or not np.isfinite(features).all() or frames.tolist()!=[r['observation']['frame_id'] for r in rows]:
            raise ValueError('Live feature alignment mismatch')
        joint_error=0.;grip_error=0.
        for step,(row,features_row) in enumerate(zip(rows,features)):
            # Reproducing the recorded proposal checks the exact pre-display inputs.
            scores=collector.learner.choose(features_row)[1]
            joints=np.clip(scores[:6]*collector.joint_gain,JOINT_LOWER,JOINT_UPPER)
            proposal=row['intervention']['policy_proposal']
            joint_error=max(joint_error,float(np.max(np.abs(joints-proposal['joints_deg']))))
            grip_error=max(grip_error,abs(neural_aperture(scores[6])-proposal['gripper_mm']))
            live[(episode['episode'],step)]=features_row
        if joint_error>.01 or grip_error>.01:
            raise ValueError('Saved live features do not reproduce the actor proposal')
        source_records.append({'episode':episode['episode'],'frames':len(rows),'split':episode['split'],
                               'joint_proposal_max_error_deg':joint_error,'gripper_proposal_max_error_mm':grip_error,
                               'live_features_sha256':manifest['live_features_sha256']})
    features=np.asarray([live[(e['episode'],e['step'])] if e['episode'] in new_names else old_lookup[(e['episode'],e['step'])] for e in examples],dtype=np.float32)
    indices={s:[i for i,e in enumerate(examples) if e['split']==s] for s in ['train','validation','test']}
    y=torch.tensor(np.array([e['target'] for e in examples]),dtype=torch.float64)
    with torch.no_grad():
        position,rotation=demonstration_target_poses(ToolKinematics(),y[:,:6],examples,meta['desired_lift_clearance_mm'],meta['lift_goal_offset_mm'],
                                                   meta['grasp_goal_height_mm'],meta['hold_center_mm'],meta['approach_center_mm'])
    weights=demonstration_weights(examples,indices['train'],meta['initial_frame_weight'],meta['first_frame_weight'],meta['grasp_frame_weight'],meta['sample_weight_policy'],meta['policy_visited_weight'])
    metadata=dict(meta)
    for key in ['metrics','stage_metrics','initial_pose_metrics','selection_statistics','continuation','sensory_cache_details','sensory_feature_cache_key']:
        metadata.pop(key,None)
    metadata.update(datasets=datasets,episodes=episodes,optimizer_iterations=0,optimizer_message='Prepared inputs only; weights unchanged',
                    training_device='prepared on Mac; not yet fitted',closed_loop_validated=False,
                    training_feature_sources=['saved normalized features with inherited source provenance','captured live MLX normalized features'],
                    captured_feature_episodes=[r['episode'] for r in source_records],
                    feature_provenance={'source_checkpoint':str(source.resolve()),'source_model_sha256':sha256_file(source/'model.npz'),
                                        'collection':str(dataset.resolve()),'collection_checkpoint':str(collection_source.resolve()),'collection_model_sha256':sha256_file(collection_source/'model.npz'),'feature_checkpoint':str(feature_source.resolve()),'live_sequences':source_records},
                    feature_adaptation_refitted=False,
                    optimizer_preconditioning='RMS output Jacobian scaling retained from source checkpoint',
                    sensory_cache_details={'kind':'saved_and_live_normalized_features','saved_frames':len(old_lookup),'live_frames':len(live)},
                    sensory_feature_cache_key=canonical_hash({'saved_features':sha256_file(feature_source/'training-features.npz'),'live_sequences':source_records}),
                    preparation_only=True)
    save_checkpoint(a.output,policy.core,policy.learner,metadata)
    np.savez(a.output/'training-features.npz',features=features)
    np.savez(a.output/'training-targets.npz',motor_targets=y.numpy(),goal_position_mm=position.numpy(),goal_rotation=rotation.numpy(),sample_weights=weights,
             preserve_expert_target=np.array([e.get('preserve_expert_target',False) for e in examples]))
    write_json(a.output/'feature-rows.json',[{'episode':e['episode'],'step':e['step'],'split':e['split'],'stage':e['stage']} for e in examples])
    write_json(a.output/'training-history.json',[])
    shutil.copy2(source/'parameter-scaling.npy',a.output/'parameter-scaling.npy')
    saved=json.loads((a.output/'manifest.json').read_text());saved.pop('files',None);artifact_manifest(a.output,saved)
    with np.load(a.output/'model.npz',allow_pickle=False) as new,np.load(source/'model.npz',allow_pickle=False) as old:
        if set(new.files)!=set(old.files) or any(not np.array_equal(new[k],old[k]) for k in old.files):
            raise AssertionError('Data preparation changed the actor')
    result={'input_checkpoint':str(a.output.resolve()),'new_verified_episodes':counts,'frames':{s:len(v) for s,v in indices.items()},
            'captured_sequences':source_records,'original_weights_unchanged':True,'preparation_only':True}
    write_json(report/'prepared-inputs.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
