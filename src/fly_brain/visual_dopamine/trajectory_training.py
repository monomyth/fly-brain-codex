"""Success-gated demonstration bootstrap through the fixed MaleCNS motor circuit.

The supervision is explicit. No teacher, geometry oracle or IK runs in the actor.
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from ..assets import home,write_json,sha256_file
from .core import VisualCore
from .motor_circuit import MotorCircuit
from .motor_core import MotorLearner
from .motor_training import save_checkpoint


def load_demonstrations(dataset,configurations,control_hz=None,grasp_tolerance_mm=None,goal_supervision="recorded",sample_hz=None,grasp_goal_height_mm=None,recovery_target_policy="recorded"):
    if goal_supervision not in ("recorded","observable","aligned"):raise ValueError("Unknown goal supervision")
    if sample_hz is not None and (not np.isfinite(sample_hz) or not 0<sample_hz<=30):raise ValueError('Sample rate must be positive and at most 30 Hz')
    if recovery_target_policy not in ("recorded","reactive"):raise ValueError("Unknown recovery target policy")
    if recovery_target_policy=="reactive" and goal_supervision!="aligned":raise ValueError("Reactive recovery requires aligned goal supervision")
    examples=[];summaries=[]
    groups={tuple(x):name for name,positions in configurations.items() if name in ('train','validation','test') for x in positions}
    episode_splits=configurations.get('episode_splits',{})
    datasets=dataset if isinstance(dataset,(list,tuple)) else [dataset]
    manifests=sorted(p for d in datasets for p in Path(d).glob('episode-*/manifest.json'))
    for manifest in manifests:
        meta=json.loads(manifest.read_text());key=tuple(meta['task']['cube_xy_mm'])
        if not(meta.get('complete') and meta.get('teacher_target_recorded')) or key not in groups:continue
        rows=[json.loads(line) for line in (manifest.parent/'steps.jsonl').read_text().splitlines()]
        native_hold=verified_native_hold(rows,meta['task'])
        contract=meta.get('supervision_contract')
        if contract not in (None,'recorded-recovery-v1'):raise ValueError('Unknown recovery supervision contract')
        recovery_episode=contract=='recorded-recovery-v1'
        if recovery_episode and (meta.get('provenance')!='teacher_assisted' or not native_hold):continue
        if not(meta.get('success') or native_hold):continue
        approach=next((r['expert_target'] for r in rows if r['stage']=='approach' and r.get('expert_target')),None)
        grasp=next((r['expert_target'] for r in rows if r['stage']=='descend' and r.get('expert_target')),None)
        transport=next((r['expert_target'] for r in rows if r['stage'] in ('lift','hold') and r.get('expert_target')),None)
        if recovery_episode and recovery_target_policy=='reactive' and grasp is None:
            contact=next((r for r in rows if all(r['observation']['finger_contacts'].values())),None)
            if contact is not None:grasp={'joints_deg':contact['observation']['joints_deg'],'gripper_mm':90.}
        if (not recovery_episode or recovery_target_policy=='reactive') and goal_supervision in ('observable','aligned') and any(x is None for x in (approach,grasp,transport)):raise ValueError('Observable-goal supervision requires complete direct-goal references')
        count=0;last_time=-np.inf;previous_stage=None
        final_step=max((i for i,r in enumerate(rows) if r['stage'] not in ['release','done']),default=-1)
        for i,row in enumerate(rows):
            if row['stage'] in ['release','done']:continue
            now=row['observation'].get('simulation_time',i/meta.get('hz',5))
            if not recovery_episode and sample_hz is not None and row['stage']==previous_stage and now-last_time<1/sample_hz-1e-6 and i!=final_step:continue
            last_time=now;previous_stage=row['stage']
            preserve_target=recovery_episode and retain_recovery_target(row,recovery_target_policy)
            target=row.get('expert_target')
            if target is None:raise ValueError('Full coordinated teacher targets are required')
            obs=row['observation']
            goal_role='grasp' if row['stage'] in ('descend','close') else 'transport' if row['stage'] in ('lift','hold') else 'approach'
            if not preserve_target and goal_supervision in ('observable','aligned'):
                path=row.get('expert_path')
                row_approach={'joints_deg':path.get('approach',approach['joints_deg']),'gripper_mm':90.} if path else approach
                row_grasp={'joints_deg':path.get('descend',grasp['joints_deg']),'gripper_mm':90.} if path else grasp
                row_transport={'joints_deg':path.get('lift',transport['joints_deg']),'gripper_mm':0.} if path else transport
                target=observable_goal_target(row,meta['task'],row_approach,row_grasp,row_transport,require_lateral_alignment=goal_supervision=='aligned',grasp_goal_height_mm=grasp_goal_height_mm,grasp_yaw_deg=meta['task'].get('cube_yaw_deg',0) if recovery_episode and recovery_target_policy=='reactive' else None)
                goal_role='grasp' if np.array_equal(target['joints_deg'],row_grasp['joints_deg']) else 'approach' if np.array_equal(target['joints_deg'],row_approach['joints_deg']) else 'transport'
            grip_target=float(target['gripper_mm']>45)
            if not preserve_target and grasp_tolerance_mm is not None and row['stage'] in ('pregrasp','descend','close'):
                from ..teacher import grasp_pose
                from scipy.spatial.transform import Rotation
                cube=row['evaluation']['cube_pose']
                yaw=Rotation.from_quat(cube['quaternion_xyzw']).as_euler('xyz',degrees=True)[2]
                desired=grasp_pose(cube['position_mm'][:2],meta['task']['cube_size_mm'],yaw)
                distance=np.linalg.norm(np.asarray(obs['grasp_pose']['position_mm'])-desired['position_mm'])
                angle=(Rotation.from_quat(desired['quaternion_xyzw']).inv()*Rotation.from_quat(obs['grasp_pose']['quaternion_xyzw'])).magnitude()
                if distance<grasp_tolerance_mm and angle<.15:grip_target=0.

            for image in obs['images']:
                if sha256_file(manifest.parent/image['file'])!=image['sha256']:raise ValueError('Camera checksum mismatch')
            examples.append({'observation':obs,'directory':manifest.parent,'target':np.r_[target['joints_deg'],grip_target].astype(np.float32),
                             'delta_target':np.r_[expert_delta(target['joints_deg'],obs['joints_deg'],control_hz or meta.get('control_hz',meta.get('hz',2.))),grip_target].astype(np.float32),
                             'recovery_episode':recovery_episode,'policy_visited':bool((row.get('intervention') or {}).get('policy_selected',False)),'grasp_pose_override':recovery_grasp_pose(row,meta['task'],grasp_goal_height_mm) if recovery_episode and recovery_target_policy=='reactive' and not preserve_target else None,'preserve_expert_target':preserve_target,'goal_role':goal_role,'gripper_relabelled':bool(grip_target!=float(target['gripper_mm']>45)),'split':episode_splits.get(manifest.parent.name,groups[key]),'stage':row['stage'],'episode':manifest.parent.name,'step':i,'reference_lift_clearance_mm':float(meta['task'].get('lift_clearance_mm',100)+15)})
            count+=1
        summaries.append({'episode':manifest.parent.name,'supervision_contract':contract,'recorded_target_preserved':recovery_episode and recovery_target_policy=='recorded','recovery_target_policy':recovery_target_policy if recovery_episode else None,'split':episode_splits.get(manifest.parent.name,groups[key]),'frames':count,'cube_xy_mm':list(key),'hold_success':bool(meta['result'].get('hold_success') or native_hold),'release_success':bool(meta['result'].get('release_success')),'source_episode_success':bool(meta['success']),'execution_mode':meta.get('execution_mode','bounded_deltas')})
    return examples,summaries


def retain_recovery_target(row,policy):
    """Retain local escape corrections, without imposing a hidden waypoint phase."""
    if policy=='recorded':return True
    # Wide-workspace examples lift above the actual grip instead of dragging to center.
    if (row.get('intervention') or {}).get('hold_in_place') and row['stage'] in ('lift','hold'):return True
    if row['stage']=='recovery_open':return True
    if row['stage']=='recovery_align':
        obs=row['observation'];cube=obs.get('cube_pose',row['evaluation']['cube_pose'])
        return np.linalg.norm(np.asarray(obs['grasp_pose']['position_mm'][:2])-cube['position_mm'][:2])>5
    return False


def recovery_grasp_pose(row,task,height):
    from ..teacher import grasp_pose
    obs=row['observation'];cube=obs.get('cube_pose',row['evaluation']['cube_pose'])
    pose=grasp_pose(cube['position_mm'][:2],task['cube_size_mm'],task.get('cube_yaw_deg',0))
    if height is not None:pose['position_mm'][2]=height
    return pose


def observable_goal_target(row,task,approach,grasp,transport,require_lateral_alignment=False,grasp_goal_height_mm=None,grasp_yaw_deg=None):
    """Training-only goal labels use observable alignment/touch instead of a hidden waypoint index.

    Reference poses come from the verified conventional demonstrations. Geometry
    is used only to supervise the camera/body controller, never during inference.
    """
    from ..teacher import grasp_pose
    from scipy.spatial.transform import Rotation
    observation=row['observation'];evaluation=row['evaluation'];contacts=observation['finger_contacts']
    if all(contacts.values()) or (any(contacts.values()) and evaluation.get('clearance_mm',0)>5):
        goal=row['expert_target'] if row['stage'] in ('lift','hold') else transport
        return {**goal,'gripper_mm':0.}
    cube=observation.get('cube_pose',evaluation['cube_pose']) if require_lateral_alignment else evaluation['cube_pose'];yaw=Rotation.from_quat(cube['quaternion_xyzw']).as_euler('xyz',degrees=True)[2]
    desired=grasp_pose(cube['position_mm'][:2],task['cube_size_mm'],yaw if grasp_yaw_deg is None else grasp_yaw_deg)
    if grasp_goal_height_mm is not None:desired['position_mm'][2]=grasp_goal_height_mm
    angle=(Rotation.from_quat(desired['quaternion_xyzw']).inv()*Rotation.from_quat(observation['grasp_pose']['quaternion_xyzw'])).magnitude()
    if angle>.15:return {**approach,'gripper_mm':90.}
    delta=np.asarray(observation['grasp_pose']['position_mm'])-desired['position_mm']
    distance=np.linalg.norm(delta)
    close=distance<12 or any(contacts.values())
    if require_lateral_alignment:
        # Training-only correction: vertical clearance may be 12 mm, but closing
        # a 20 mm cube grip with a 10 mm sideways miss pushes the cube away.
        close=close and np.linalg.norm(delta[:2])<min(5.,task.get('cube_size_mm',20)*.25)
    return {**grasp,'gripper_mm':0. if close else 90.}


def verified_native_hold(rows,task):
    """Admit a completed pickup/hold even if a later release/cleanup failed."""
    for row in rows:
        e=row.get('evaluation') or {}
        if (e.get('success') and e.get('held') and e.get('episode_id')==row['observation'].get('episode_id')
            and e.get('hold_seconds',0)>=task.get('hold_seconds',5)-1e-9
            and e.get('clearance_mm',0)>=task.get('lift_clearance_mm',100)
            and e.get('tilt_deg',180)<=task.get('tilt_tolerance_deg',5)):
            return True
    return False


def expert_delta(target,current,hz):
    delta=np.asarray(target)-np.asarray(current)
    return delta*min(1.,(30./hz)/max(float(np.max(np.abs(delta))),1e-9))


def torch_sparse(array):
    value=array.tocoo()
    return torch.sparse_coo_tensor(torch.tensor(np.array([value.row,value.col]),dtype=torch.long),torch.tensor(value.data,dtype=torch.float32),value.shape).coalesce()


class MotorTargetNetwork(nn.Module):
    def __init__(self,circuit,train_motor=False):
        super().__init__();self.weights=nn.Parameter(torch.from_numpy(circuit.plastic_base.copy()))
        self.excitability=nn.Parameter(torch.zeros(len(circuit.mbon)))
        self.register_buffer('base',torch.from_numpy(circuit.plastic_base.copy()))
        self.register_buffer('mask',self.base>0)
        self.register_buffer('input_signs',torch.tensor(getattr(circuit,'input_signs',np.ones(len(circuit.kc),dtype=np.float32))))
        for name in ['cb_from_mbon','dn_from_mbon_cb','pm_from_dn','motor_from_dn_pm']:
            self.register_buffer(name,torch_sparse(getattr(circuit,name)))
        self.register_buffer('readout',torch.tensor(circuit.readout));self.register_buffer('gain',torch.tensor(circuit.gain))
        learner=MotorLearner(circuit);learner.configure_centered_response()
        self.register_buffer('reference',torch.tensor(learner.reference_scores))
        self.register_buffer('rest_current',torch.tensor(learner.rest_current))
        self.register_buffer('dopamine_gain',torch.tensor(circuit.dopamine_weights@np.full(len(circuit.dan),np.tanh(.5),dtype=np.float32)))
        self.joint_gain=1440.
        self.train_motor=train_motor
        if train_motor:
            matrix=circuit.motor_from_dn_pm.tocoo()
            self.motor_strength=nn.Parameter(torch.tensor(np.abs(matrix.data)))
            self.register_buffer('motor_signs',torch.tensor(np.sign(matrix.data)))
            self.register_buffer('motor_indices',torch.tensor(np.array([matrix.row,matrix.col]),dtype=torch.long))
            self.register_buffer('motor_base',torch.tensor(np.abs(matrix.data)))
            self.motor_shape=matrix.shape

    def forward(self,features):
        rates=.1*features*self.input_signs
        mbon=torch.tanh(rates@(self.weights*self.mask).T+self.excitability)
        cb=torch.tanh(torch.sparse.mm(self.cb_from_mbon,mbon.T).T)
        dn=torch.tanh(torch.sparse.mm(self.dn_from_mbon_cb,torch.cat([mbon,cb],dim=1).T).T)
        pm=torch.tanh(torch.sparse.mm(self.pm_from_dn,dn.T).T)
        motor_matrix=(torch.sparse_coo_tensor(self.motor_indices,self.motor_strength*self.motor_signs,self.motor_shape).coalesce() if self.train_motor else self.motor_from_dn_pm)
        motor=torch.tanh(torch.sparse.mm(motor_matrix,torch.cat([dn,pm],dim=1).T).T)
        return (motor@self.readout.T)*self.gain-self.reference


def train(dataset,output,configurations,root=None,epochs=600,seed=0,learning_rate=.003,dopamine_enabled=True):
    torch.set_num_threads(2);torch.manual_seed(seed);root=home(root);output=Path(output)
    if output.exists():raise FileExistsError('Choose a new trajectory checkpoint directory')
    from .motor_circuit import prepare
    path,meta=prepare(root);circuit=MotorCircuit(path,root)
    examples,summaries=load_demonstrations(dataset,configurations)
    if not examples:raise ValueError('No successful full-target camera demonstrations')
    training=[i for i,x in enumerate(examples) if x['split']=='train'];validation=[i for i,x in enumerate(examples) if x['split']=='validation']
    if not training:raise ValueError('Training configurations are required')
    output.parent.mkdir(parents=True,exist_ok=True)
    cache=output.with_name(output.name+'-features.npz')
    core=VisualCore(circuit)
    retinal=[core.encoder.sample(e['observation']['images'],e['directory']) for e in examples]
    core.fit_adaptation([retinal[i] for i in training],local_contrast=True)
    features=np.asarray([core.encode(value) for value in retinal])
    np.savez(cache,features=features,retinal_mean=core.retinal_mean,retinal_std=core.retinal_std,kc_mean=core.kc_mean,kc_std=core.kc_std)
    x=torch.tensor(features);targets=torch.tensor(np.asarray([e['target'] for e in examples]))
    counts=Counter(examples[i]['stage'] for i in training)
    sample_weights=torch.tensor([1/counts[e['stage']] if e['stage'] in counts else 1. for e in examples],dtype=torch.float32)
    sample_weights/=sample_weights[training].mean()
    model=MotorTargetNetwork(circuit);optimizer=torch.optim.Adam([model.weights,model.excitability],lr=learning_rate)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=learning_rate*.005)
    initial=model.weights.detach().clone();best=None;best_score=float('inf');history=[]
    selection=validation or training
    def statistics(indices):
        with torch.no_grad():
            scores=model(x[indices]);errors=(scores[:,:6]*model.joint_gain-targets[indices,:6]).abs()
            return {'joint_mae_deg':errors.mean(dim=0).tolist(),'joint_rmse_deg':float(torch.sqrt((errors**2).mean())),
                    'gripper_accuracy':float(((scores[:,6]>=0)==(targets[indices,6]>=.5)).float().mean())}
    for epoch in range(epochs):
        model.train();indices=np.random.default_rng(seed+epoch).permutation(training)
        loss_total=0.
        for start in range(0,len(indices),128):
            selected=indices[start:start+128];scores=model(x[selected])
            joint_error=(scores[:,:6]*model.joint_gain-targets[selected,:6])/torch.tensor([5.,15.,15.,15.,5.,5.])
            joint_loss=(joint_error**2).mean(dim=1)
            grip_loss=nn.functional.binary_cross_entropy_with_logits(scores[:,6]*60,targets[selected,6],reduction='none')
            loss=((joint_loss+.2*grip_loss)*sample_weights[selected]).mean()
            optimizer.zero_grad();loss.backward()
            # Explicit supervised eligibility; successful episode return gates PAM01 activity.
            model.weights.grad*=model.mask*model.dopamine_gain[:,None]
            model.excitability.grad*=model.dopamine_gain
            if dopamine_enabled:
                torch.nn.utils.clip_grad_norm_([model.weights,model.excitability],10.)
                optimizer.step()
                with torch.no_grad():model.weights.clamp_(min=0);model.weights.copy_(torch.minimum(model.weights,4*model.base));model.weights*=model.mask;model.excitability.clamp_(-2,2)
            loss_total+=float(loss.detach())
        scheduler.step()
        if epoch%20==0 or epoch+1==epochs:
            train_stats=statistics(training);valid_stats=statistics(selection)
            score=valid_stats['joint_rmse_deg']+3*(1-valid_stats['gripper_accuracy'])
            history.append({'epoch':epoch+1,'train':train_stats,'validation':valid_stats,'loss':loss_total})
            print(f"Epoch {epoch+1}: train {train_stats['joint_rmse_deg']:.3f} deg, selection {valid_stats['joint_rmse_deg']:.3f} deg, grip {valid_stats['gripper_accuracy']:.3f}",flush=True)
            if score<best_score:best_score=score;best=(model.weights.detach().clone(),model.excitability.detach().clone())
    with torch.no_grad():model.weights.copy_(best[0]);model.excitability.copy_(best[1])
    if not dopamine_enabled and not torch.equal(initial,model.weights):raise AssertionError('Weights changed with PAM learning disabled')
    learner=MotorLearner(circuit,seed=seed,learning_rate=1e-7,noise=.0003);learner.configure_centered_response();learner.weights[:]=model.weights.detach().numpy();learner.excitability[:]=model.excitability.detach().numpy()
    metrics={name:statistics([i for i,e in enumerate(examples) if e['split']==name]) for name in ['train','validation','test'] if any(e['split']==name for e in examples)}
    metadata={'seed':seed,'dopamine_enabled':dopamine_enabled,'dataset':str(Path(dataset).resolve()),'episodes':summaries,'epochs':epochs,
              'training_method':'Successful-demonstration bootstrap with supervised pose-target eligibility and PAM01 gating; optional online reward fine-tuning',
              'supervised_bootstrap':True,'trainable_parameters':{'synapses':int(np.count_nonzero(circuit.plastic_base)),'mbon_excitability_offsets':len(circuit.mbon)},'bootstrap_dopamine_signal':float(np.tanh(.5)) if dopamine_enabled else 0.,
              'architecture':{'kind':'malecns_motor_dopamine','updates':12,'output_channels':7,'output_mode':'joint_targets','joint_gain':model.joint_gain,'control_hz':2.},
              'task_kind':'camera_pickup_hold','task_scope':'Image-based continuous joint target controller; independent live validation required',
              'closed_loop_validated':False,'metrics':metrics}
    save_checkpoint(output,core,learner,metadata)
    write_json(output/'training-history.json',history)
    from ..assets import artifact_manifest
    m=json.loads((output/'manifest.json').read_text());m.pop('files',None);artifact_manifest(output,m)
    return {'checkpoint':str(output),'metrics':metrics}
