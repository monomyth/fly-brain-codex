"""Supervised camera/body trajectory bootstrap through measured neural routes.

Successful episodes provide teacher targets. PAM01 activity gates optimization;
this is engineered supervised credit assignment, not pure reward-only learning.
"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from ..assets import home,write_json,artifact_manifest,canonical_hash,sha256_file
from .feedback import FeedbackCircuit,FeedbackCore,prepare
from .trajectory_training import load_demonstrations,MotorTargetNetwork
from .motor_core import MotorLearner
from .motor_training import save_checkpoint


def demonstration_weights(examples, training_indices, initial_weight=1., first_frame_weight=1., grasp_frame_weight=1.,sample_weight_policy="recorded_stage",policy_visited_weight=1.):
    """Training-only phase emphasis; normalization depends on training rows alone."""
    weights=[]
    for example in examples:
        obs=example['observation'];q=np.asarray(obs['joints_deg'])
        base=first_frame_weight if np.max(np.abs(q))<1e-6 and obs['gripper_mm']<1e-6 else initial_weight if np.max(np.abs(q[1:4]))<20 else 1.
        emphasized=example['stage'] in ('descend','close','recovery_open','recovery_align')
        if sample_weight_policy=='goal_role':emphasized=emphasized or example.get('goal_role')=='grasp'
        value=max(base,grasp_frame_weight) if emphasized else base
        if example.get('policy_visited'):value=max(value,policy_visited_weight)
        weights.append(value)
    weights=np.asarray(weights,dtype=np.float64)
    return weights/weights[training_indices].mean()


def validation_score(stats,task_space):
    if stats.get('groups'):
        return float(np.mean([validation_score(value,task_space) for value in stats['groups'].values()]))
    error=stats['goal_position_rmse_mm']/5+stats['goal_orientation_rmse_deg']/5 if task_space else stats['joint_rmse_deg']
    return error+3*(1-stats['gripper_accuracy'])


def train(dataset,output,configurations,root=None,iterations=2000,seed=0,dopamine_enabled=True,output_mode='joint_deltas',warm_start=None,device='cpu',train_motor=False,goal_supervision='recorded',precondition=False,initial_weight=1.,first_frame_weight=1.,refit_adaptation=False,retinal_profile=None,sample_hz=None,task_space_loss=False,lift_goal_offset_mm=0.,desired_lift_clearance_mm=None,grasp_frame_weight=1.,grasp_goal_height_mm=None,hold_center_mm=None,approach_center_mm=None,temporal_sensory=False,refit_feature_adaptation=False,feedback_circuit=None,transfer_readout=False,recovery_target_policy="recorded",sample_weight_policy="recorded_stage",policy_visited_weight=1.,validation_aggregation="frames"):
    root=home(root);output=Path(output)
    if output.exists():raise FileExistsError('Choose a new checkpoint directory')
    if output_mode not in ('joint_targets','joint_deltas'):raise ValueError('Unknown output mode')
    if task_space_loss and output_mode!='joint_targets':raise ValueError('Task-space supervision requires absolute joint targets')
    if not np.isfinite(lift_goal_offset_mm) or not 0<=lift_goal_offset_mm<=50 or (lift_goal_offset_mm and not task_space_loss):raise ValueError('A 0-50 mm lift target offset requires task-space supervision')
    if desired_lift_clearance_mm is not None and (not task_space_loss or lift_goal_offset_mm or not np.isfinite(desired_lift_clearance_mm) or not 100<=desired_lift_clearance_mm<=180):raise ValueError('A common 100-180 mm lift target requires task-space supervision and no additional offset')
    if grasp_goal_height_mm is not None and (not task_space_loss or not np.isfinite(grasp_goal_height_mm) or not 22<=grasp_goal_height_mm<=40):raise ValueError('A 22-40 mm grasp goal height requires task-space supervision')
    if hold_center_mm is not None and (not task_space_loss or np.asarray(hold_center_mm).shape!=(2,) or not np.isfinite(hold_center_mm).all()):raise ValueError('A finite XY hold center requires task-space supervision')
    if approach_center_mm is not None and (not task_space_loss or np.asarray(approach_center_mm).shape!=(2,) or not np.isfinite(approach_center_mm).all()):raise ValueError('A finite XY approach center requires task-space supervision')
    if iterations<1:raise ValueError('Positive iteration count required')
    if not np.isfinite([initial_weight,first_frame_weight]).all() or not 1<=initial_weight<=100 or not 1<=first_frame_weight<=100:raise ValueError('Initial-frame weights must be between 1 and 100')
    if not np.isfinite(grasp_frame_weight) or not 1<=grasp_frame_weight<=100:raise ValueError('Grasp-frame weight must be between 1 and 100')
    if transfer_readout and (warm_start is None or not train_motor):raise ValueError('Readout transfer requires a warm start and motor-synapse training')
    if sample_weight_policy not in ("recorded_stage","goal_role"):raise ValueError("Unknown sample weighting policy")
    if validation_aggregation not in ("frames","recovery_balanced"):raise ValueError("Unknown validation aggregation")
    if not np.isfinite(policy_visited_weight) or not 1<=policy_visited_weight<=100:raise ValueError("Policy-visited weight must be between 1 and 100")
    torch.set_num_threads(2);torch.manual_seed(seed)
    if feedback_circuit is None:path,meta=prepare(root)
    else:
        path=Path(feedback_circuit);meta=json.loads((path/'manifest.json').read_text())
    circuit=FeedbackCircuit(path,root)
    if temporal_sensory:
        from .temporal import TemporalFeedbackCore
        core=TemporalFeedbackCore(circuit,config=temporal_sensory if isinstance(temporal_sensory,dict) else None)
    else:core=FeedbackCore(circuit)
    examples,episodes=load_demonstrations(dataset,configurations,control_hz=2.,goal_supervision=goal_supervision,sample_hz=sample_hz,grasp_goal_height_mm=grasp_goal_height_mm,recovery_target_policy=recovery_target_policy)
    indices={name:[i for i,e in enumerate(examples) if e['split']==name] for name in ['train','validation','test']}
    if not indices['train']:raise ValueError('Successful training trajectories are required')
    if (configurations.get('validation') or 'validation' in configurations.get('episode_splits',{}).values()) and not indices['validation']:raise ValueError('The requested validation split has no verified trajectories')
    print({'frames':{k:len(v) for k,v in indices.items()},'plastic_edges':meta['plastic_edges']},flush=True)
    old=None;transfer_details=None
    if warm_start:
        from .motor_policy import MotorPolicy
        old=MotorPolicy(warm_start,root)
        if old.metadata.get('feedback_circuit_id')!=meta['feedback_circuit_id']:
            if not transfer_readout:raise ValueError('Warm start circuit mismatch')
            from .ltm_circuit import validate_transfer
            transfer_details=validate_transfer(old.circuit,circuit)
        if old.metadata.get('temporal_sensory')!=getattr(core,'temporal_config',None):refit_feature_adaptation=True
    from ..cameras import observation_contract
    contracts={observation_contract(e['observation']) for e in examples}
    if len(contracts)!=1:raise ValueError('Training cannot mix camera rigs; collect a consistent Front/Gripper dataset')
    camera_names, camera_revision=next(iter(contracts))
    core.encoder.configure_cameras(camera_names,camera_revision)
    changed_cameras=old is not None and (camera_names,camera_revision)!=(old.core.encoder.camera_names,old.core.encoder.camera_rig_revision)
    selected_profile=retinal_profile if retinal_profile is not None else (old.core.encoder.profile if old and not changed_cameras else None)
    core.encoder.configure(selected_profile)
    if old is not None and (changed_cameras or core.encoder.profile!=old.core.encoder.profile):refit_adaptation=True
    retinal=np.asarray([core.encoder.sample(e['observation']['images'],e['directory']) for e in examples])
    body=np.asarray([core.body_encoder.sample(e['observation']) for e in examples])
    keep_adaptation=old is not None and not refit_adaptation
    if keep_adaptation:
        for name in ['retinal_mean','retinal_std','kc_mean','kc_std']:setattr(core,name,getattr(old.core,name).copy())
    else:
        core.retinal_mean=retinal[indices['train']].mean(axis=0)
        core.retinal_std=np.maximum(retinal[indices['train']].std(axis=0),.01)
    if temporal_sensory:
        from .temporal import sequence_responses
        def temporal_progress(done,total):print(f'Temporal encoding {done}/{total} observations',flush=True)
        raw,cache_statistics=sequence_responses(core,retinal,body,examples,root/'feature-cache/temporal-v1',temporal_progress)
        cache_key=canonical_hash(cache_statistics['sequences'])
        print({'sensory_feature_cache':cache_statistics},flush=True)
    else:
        from .feedback_cache import calibration_recipe,cached_responses
        cache_core=old.core if transfer_details and keep_adaptation and old.metadata.get('temporal_sensory')==getattr(core,'temporal_config',None) else core
        recipe=calibration_recipe(cache_core,device)
        cache_key=canonical_hash({'schema':2,**recipe,'retinal':canonical_hash(retinal.tolist()),'body':canonical_hash(body.tolist()),'train_indices':indices['train']})
        cache=root/'feature-cache/feedback'/f'{cache_key}.npz';cache.parent.mkdir(parents=True,exist_ok=True)
        precomputed=None
        if cache.exists():
            with np.load(cache,allow_pickle=False) as saved:precomputed=saved['raw_responses'].copy()
        def encode_missing(retinal_values,body_values):
            if device.startswith('cuda'):
                from .feedback import TorchFeedbackEncoder
                return TorchFeedbackEncoder(core,device).responses(retinal_values,body_values)
            from .feedback import ScheduledFeedbackEncoder
            encoder=ScheduledFeedbackEncoder(core)
            def progress(done,total):
                if done%32==0 or done==total:print(f'Encoded {done}/{total} new camera/body observations',flush=True)
            return encoder.responses(retinal_values,body_values,progress=progress)
        row_cache=root/'feature-cache/feedback/rows'/f"{canonical_hash({'schema':1,**recipe})}.npz"
        raw,cache_statistics=cached_responses(row_cache,retinal,body,len(circuit.kc),encode_missing,precomputed)
        print({'sensory_feature_cache':cache_statistics,'path':str(row_cache)},flush=True)
        if not cache.exists():
            temporary=cache.with_suffix('.partial.npz')
            np.savez_compressed(temporary,raw_responses=raw);temporary.replace(cache)
    if not keep_adaptation or refit_feature_adaptation:
        core.kc_mean=raw[indices['train']].mean(axis=0);std=raw[indices['train']].std(axis=0)
        core.kc_std=np.maximum(std,max(float(std.max())*.0005,1e-12))
    features=core.encode_responses(raw)
    if features.shape!=(len(examples),len(circuit.kc)) or not np.isfinite(features).all():raise ValueError('Invalid sensory feature cache')
    x=torch.tensor(features,dtype=torch.float64,device=device)
    y=torch.tensor(np.array([e['delta_target' if output_mode=='joint_deltas' else 'target'] for e in examples]),dtype=torch.float64,device=device)
    sample_weights=torch.tensor(demonstration_weights(examples,indices['train'],initial_weight,first_frame_weight,grasp_frame_weight,sample_weight_policy,policy_visited_weight),dtype=torch.float64,device=device)
    model=MotorTargetNetwork(circuit,train_motor=train_motor).double().to(device);post,pre=np.nonzero(circuit.plastic_base);n=len(post);mb=len(circuit.mbon)
    start=np.r_[.01*circuit.plastic_base[post,pre],np.zeros(mb)].astype(float)
    if old is not None:
        start=np.r_[old.learner.weights[post,pre],old.learner.excitability].astype(float)
    bounds=list(zip(np.zeros(n),64*circuit.plastic_base[post,pre]))+[(-2.,2.)]*mb
    if train_motor:
        if transfer_details:
            from .ltm_circuit import transferred_strengths
            motor_initial=transferred_strengths(old.circuit,circuit)
        else:motor_initial=np.abs(old.circuit.motor_from_dn_pm.data if old is not None else circuit.motor_from_dn_pm.data)
        start=np.r_[start,motor_initial]
        bounds+=list(zip(np.zeros(len(motor_initial)),16*np.abs(circuit.motor_from_dn_pm.data)))
    fk=None;target_position=None;target_rotation=None
    if task_space_loss:
        from ..kinematics import ToolKinematics,demonstration_target_poses
        fk=ToolKinematics().double().to(device)
        grasp_rows=torch.tensor([e.get('goal_role')=='grasp' for e in examples],dtype=torch.bool,device=device)
        with torch.no_grad():
            target_position,target_rotation=demonstration_target_poses(fk,y[:,:6],examples,desired_lift_clearance_mm,lift_goal_offset_mm,grasp_goal_height_mm,hold_center_mm,approach_center_mm)
    selection=indices['validation'] or indices['train'];history=[];calls=0
    best_value=float('inf');best=None

    def assign(v):
        with torch.no_grad():
            model.weights.zero_();model.weights[post,pre]=torch.tensor(v[:n],device=device);model.excitability.copy_(torch.tensor(v[n:n+mb],device=device))
            if train_motor:model.motor_strength.copy_(torch.tensor(v[n+mb:],device=device))

    def statistics(rows):
        with torch.no_grad():
            s=model(x[rows]);error=s[:,:6]*model.joint_gain-y[rows,:6]
            stats={'joint_rmse_deg':float(torch.sqrt(error.square().mean())), 'joint_mae_deg':error.abs().mean(0).tolist(),
                    'gripper_accuracy':float(((s[:,6]>=0)==(y[rows,6]>.5)).double().mean())}
            if fk is not None:
                position,rotation=fk(s[:,:6]*model.joint_gain)
                delta=position-target_position[rows]
                angular=(rotation-target_rotation[rows]).square().sum((1,2))/2*(180/torch.pi)**2
                stats.update(goal_position_rmse_mm=float(torch.sqrt(delta.square().sum(1).mean())),goal_orientation_rmse_deg=float(torch.sqrt(angular.mean())))
            return stats

    def selection_statistics():
        stats=statistics(selection)
        if validation_aggregation=='recovery_balanced':
            groups={name:[i for i in selection if bool(examples[i].get('recovery_episode'))==recovery] for name,recovery in [('legacy',False),('recovery',True)]}
            stats['groups']={name:statistics(rows) for name,rows in groups.items() if rows}
        return stats

    def selection_value(stats):
        return validation_score(stats,fk is not None)

    def objective(v):
        nonlocal calls,best,best_value
        assign(v);model.zero_grad();s=model(x[indices['train']]);target=y[indices['train']]
        joint_rows=(s[:,:6]*model.joint_gain-target[:,:6]).square().mean(dim=1)
        grip_rows=torch.nn.functional.binary_cross_entropy_with_logits(s[:,6]*2000,target[:,6],reduction='none')
        joint=(joint_rows*sample_weights[indices['train']]).mean();grip=(grip_rows*sample_weights[indices['train']]).mean()
        # Successful demonstration return=1, tonic reward expectation=.5.
        regression=joint
        if fk is not None:
            position,rotation=fk(s[:,:6]*model.joint_gain)
            distance=(position-target_position[indices['train']]).square().sum(1)
            if grasp_goal_height_mm is not None:
                undershoot=torch.relu(grasp_goal_height_mm-3.-position[:,2])
                distance=distance+4*undershoot.square()*grasp_rows[indices['train']]
            angle=(rotation-target_rotation[indices['train']]).square().sum((1,2))/2*(180/torch.pi)**2
            # Normalize position and orientation to 5 mm / 5 degree task tolerances.
            regression=.04*joint+((distance+angle)/25*sample_weights[indices['train']]).mean()
        loss=(regression+2*grip)*(np.tanh(.5) if dopamine_enabled else 0.)
        loss.backward();gradient=np.r_[model.weights.grad.cpu().numpy()[post,pre],model.excitability.grad.cpu().numpy()].astype(float)
        if train_motor:gradient=np.r_[gradient,model.motor_strength.grad.cpu().numpy()]
        calls+=1
        if calls==1 or calls%100==0:
            stats=selection_statistics();value=selection_value(stats)
            if value<best_value:best_value=value;best=v.copy()
            row={'calls':calls,'train_joint_rmse_deg':float(torch.sqrt(joint).detach()),'selection':stats};history.append(row);print(row,flush=True)
        return float(loss.detach()),gradient

    parameter_scale=np.ones_like(start)
    if precondition:
        assign(start)
        squared=np.zeros_like(start)
        probes=[indices['train'][i] for i in np.linspace(0,len(indices['train'])-1,min(12,len(indices['train'])),dtype=int)]
        for index in probes:
            for axis in range(7):
                model.zero_grad();scalar=model(x[index:index+1])[0,axis]*(model.joint_gain if axis<6 else 2000.)
                scalar.backward()
                gradient=np.r_[model.weights.grad.cpu().numpy()[post,pre],model.excitability.grad.cpu().numpy()]
                if train_motor:gradient=np.r_[gradient,model.motor_strength.grad.cpu().numpy()]
                squared+=gradient**2
        parameter_scale=np.clip(np.sqrt(squared/(len(probes)*7)),1e-3,1e5)
        print({'preconditioning_probes':len(probes),'scale_min':float(parameter_scale.min()),'scale_max':float(parameter_scale.max())},flush=True)
    def scaled_objective(value):
        loss,gradient=objective(value/parameter_scale)
        return loss,gradient/parameter_scale
    scaled_bounds=[(a*s,b*s) for (a,b),s in zip(bounds,parameter_scale)]
    result=minimize(scaled_objective,start*parameter_scale,jac=True,bounds=scaled_bounds,method='L-BFGS-B',options={'maxiter':iterations,'ftol':1e-11,'gtol':1e-7,'maxls':40,'maxcor':50})
    result.x=result.x/parameter_scale
    assign(result.x);last=selection_statistics();value=selection_value(last)
    chosen=result.x if value<=best_value else best
    if not dopamine_enabled:
        if not np.allclose(chosen,start,rtol=1e-14,atol=1e-14):raise AssertionError('Parameters changed with dopamine disabled')
        chosen=start.copy()
    assign(chosen);metrics={k:statistics(v) for k,v in indices.items() if v}
    stage_metrics={name:{stage:statistics([i for i in rows if examples[i]['stage']==stage]) for stage in sorted({examples[i]['stage'] for i in rows})} for name,rows in indices.items() if rows}
    initial_metrics={}
    for name,rows in indices.items():
        selected=[i for i in rows if max(abs(np.asarray(examples[i]['observation']['joints_deg'])))<1e-6 and examples[i]['observation']['gripper_mm']<1e-6]
        if selected:initial_metrics[name]=statistics(selected)
    learner=MotorLearner(circuit,seed=seed,learning_rate=1e-7,noise=.0003);learner.configure_centered_response();learner.weight_multiple=64.
    learner.weights[:]=model.weights.detach().cpu().numpy();learner.excitability[:]=model.excitability.detach().cpu().numpy()
    if train_motor:circuit.motor_from_dn_pm.data=(model.motor_strength*model.motor_signs).detach().cpu().numpy().astype(np.float32)
    direct_execution=output_mode=='joint_targets' and all(e['execution_mode']=='absolute_targets' for e in episodes)
    metadata={'sample_weight_policy':sample_weight_policy,'policy_visited_weight':policy_visited_weight,'validation_aggregation':validation_aggregation,'selection_statistics':selection_statistics(),'recovery_target_policy':recovery_target_policy,'readout_transfer':transfer_details,'temporal_sensory':getattr(core,'temporal_config',None),'feature_adaptation_refitted':bool(not keep_adaptation or refit_feature_adaptation),'sensory_cache_details':cache_statistics,'approach_center_mm':list(approach_center_mm) if approach_center_mm is not None else None,'grasp_label_height_mm':grasp_goal_height_mm,'hold_center_mm':list(hold_center_mm) if hold_center_mm is not None else None,'grasp_goal_height_mm':grasp_goal_height_mm,'grasp_undershoot_penalty':4 if grasp_goal_height_mm is not None else 0,'grasp_frame_weight':grasp_frame_weight,'stage_metrics':stage_metrics,'desired_lift_clearance_mm':desired_lift_clearance_mm,'lift_goal_offset_mm':lift_goal_offset_mm,'joint_metrics_note':'Deviation from original teacher posture; tool-position targets include the declared grasp/lift adjustments' if lift_goal_offset_mm or desired_lift_clearance_mm is not None or grasp_goal_height_mm is not None else None,'task_space_loss':task_space_loss,'task_space_scales':{'position_mm':5,'orientation_deg':5,'joint_posture_weight':.04} if task_space_loss else None,'sample_hz':sample_hz,'sensory_feature_cache_key':cache_key,'adaptation_refitted':refit_adaptation,'first_frame_weight':first_frame_weight,'initial_pose_metrics':initial_metrics,'initial_frame_weight':initial_weight,'optimizer_preconditioning':'RMS output Jacobian on training probes' if precondition else 'none','goal_supervision':goal_supervision,'reference_execution_mode':'absolute_targets' if direct_execution else 'bounded_deltas','seed':seed,'dopamine_enabled':dopamine_enabled,'datasets':[str(Path(d).resolve()) for d in (dataset if isinstance(dataset,(list,tuple)) else [dataset])],'episodes':episodes,
        'training_device':device,'motor_synapse_plasticity':train_motor,'supervised_bootstrap':True,'training_method':'Successful teacher-trajectory imitation; PAM01-gated supervised objective, constrained L-BFGS-B optimization with PyTorch autograd',
        'architecture':{'kind':'malecns_motor_dopamine','updates':12,'output_channels':7,'output_mode':output_mode,'joint_gain':model.joint_gain,'control_hz':2.,'ramp_gripper':not direct_execution,'settle_before_observation':True,'position_servo':output_mode=='joint_targets' and not direct_execution,'command_scale':.5 if output_mode=='joint_targets' and not direct_execution else 1.},
        'observation_schema':{'mode':'vision','retinal_inputs':'/'.join(core.encoder.camera_names)+' luminance to R1-R6','body_inputs':['joints_deg','gripper_mm','finger_contacts'],'privileged_cube_pose':False},
        'task_kind':'camera_body_pickup_hold','task_scope':'Camera and proprioceptive neural controller; autonomous pickup requires independent live validation','closed_loop_validated':False,
        'trainable_parameters':{'existing_synapses':n,'mbon_excitability_offsets':mb,'motor_synapses':len(circuit.motor_from_dn_pm.data) if train_motor else 0},'metrics':metrics,'optimizer_iterations':int(result.nit),'optimizer_message':str(result.message),
        'modelling_assumptions':['Engineering rate-deviation model; not a validated biological simulation','Robot encoders and finger contacts stimulate recorded leg afferents using declared population tuning','Existing visual KC and ascending-neuron inputs to MBONs learn; measured wiring and muscle-pool mapping stay fixed; final motor strengths can also learn','MBON excitability offsets also learn; anatomical contact counts are an initialization prior with a declared 64x bound','Supervised teacher targets supply credit; PAM01 gates learning','Optional final motor-synapse training preserves measured edges and transmitter signs; global reward gating there is an engineering assumption, not established PAM01 innervation','No teacher, IK, cube XYZ, task phase, or reward state enters actor inference']}
    if task_space_loss:
        from ..kinematics import DEFINITION
        metadata['robot_kinematics_sha256']=sha256_file(DEFINITION)
    save_checkpoint(output,core,learner,metadata);write_json(output/'training-history.json',history)
    np.savez(output/'training-features.npz',features=features)
    targets={'sample_weights':sample_weights.detach().cpu().numpy(),'motor_targets':y.detach().cpu().numpy(),'preserve_expert_target':np.asarray([e.get('preserve_expert_target',False) for e in examples])}
    if target_position is not None:targets.update(goal_position_mm=target_position.detach().cpu().numpy(),goal_rotation=target_rotation.detach().cpu().numpy())
    np.savez(output/'training-targets.npz',**targets)
    np.save(output/'parameter-scaling.npy',parameter_scale)
    write_json(output/'feature-rows.json',[{'episode':e['episode'],'step':e['step'],'split':e['split'],'stage':e['stage']} for e in examples])
    metadata=json.loads((output/'manifest.json').read_text());metadata.pop('files');artifact_manifest(output,metadata)
    return {'checkpoint':str(output),'metrics':metrics}
