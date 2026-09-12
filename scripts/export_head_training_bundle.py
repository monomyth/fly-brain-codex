"""Export a verified, compact continuation-training snapshot for Blacktower."""
import argparse
import copy
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from fly_brain.assets import write_json,sha256_file,artifact_manifest
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.trajectory_training import MotorTargetNetwork,load_demonstrations
from probability_gripper import neural_aperture

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--configurations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--held-orientation-objective',choices=['full_rotation','cube_up'],default='full_rotation');p.add_argument('--pregrasp-retention-strength',type=float,default=0.);p.add_argument('--retention-selection-group',choices=['all','legacy'],default='all');p.add_argument('--retention-joint-rmse-limit',type=float,default=.1);p.add_argument('--retention-joint-max-limit',type=float,default=.5);p.add_argument('--retention-aperture-limit',type=float,default=1.);p.add_argument('--retention-training-group',choices=['all','legacy'],default='all');p.add_argument('--held-position-objective',choices=['tool_position','clearance'],default='tool_position');p.add_argument('--gripper-loss-weight',type=float,default=2.);p.add_argument('--precision-validation-weight',type=float,default=0.);p.add_argument('--training-scope',choices=['all','held','unheld'],default='all');p.add_argument('--orientation-loss-weight',type=float,default=1.);a=p.parse_args()
if not np.isfinite(a.orientation_loss_weight) or not 0<a.orientation_loss_weight<=100:raise ValueError('Invalid orientation loss weight')
if not np.isfinite(a.precision_validation_weight) or not 0<=a.precision_validation_weight<=16:raise ValueError('Invalid validation precision weight')
if not np.isfinite(a.gripper_loss_weight) or a.gripper_loss_weight<=0:raise ValueError('Gripper loss weight must be positive')
if a.held_position_objective=='clearance' and a.held_orientation_objective!='cube_up':raise ValueError('Clearance loss requires cube-up supervision')
if not np.isfinite(a.pregrasp_retention_strength) or not 0<=a.pregrasp_retention_strength<=100:raise ValueError('Invalid pre-grasp retention strength')
if not (0<a.retention_joint_rmse_limit<=1 and a.retention_joint_rmse_limit<=a.retention_joint_max_limit<=2 and 0<a.retention_aperture_limit<=5):raise ValueError('Invalid bounded retention limits')
root=Path(__file__).resolve().parents[1]
if a.output.exists() or not a.output.resolve().is_relative_to(root):raise ValueError('Use a new project-local bundle directory')
torch.set_num_threads(2)
policy=MotorPolicy(a.checkpoint,root/'data');meta=policy.metadata
circuit=copy.copy(policy.circuit);circuit.motor_from_dn_pm=policy.circuit.motor_synapse_base.copy()
model=MotorTargetNetwork(circuit,train_motor=True).double();model.joint_gain=policy.joint_gain
with torch.no_grad():
    model.weights.copy_(torch.tensor(policy.learner.weights,dtype=torch.float64))
    model.excitability.copy_(torch.tensor(policy.learner.excitability,dtype=torch.float64))
    model.motor_strength.copy_(torch.tensor(np.abs(policy.circuit.motor_from_dn_pm.data),dtype=torch.float64))
rows=json.loads((a.checkpoint/'feature-rows.json').read_text())
examples,_=load_demonstrations(meta['datasets'],json.loads(a.configurations.read_text()),goal_supervision=meta['goal_supervision'],sample_hz=meta['sample_hz'],grasp_goal_height_mm=meta['grasp_goal_height_mm'],recovery_target_policy=meta['recovery_target_policy'])
assert [(r['episode'],r['step'],r['split']) for r in rows]==[(e['episode'],e['step'],e['split']) for e in examples]
with np.load(a.checkpoint/'training-features.npz',allow_pickle=False) as f:features=f['features'].copy()
with np.load(a.checkpoint/'training-targets.npz',allow_pickle=False) as f:targets={k:f[k].copy() for k in f.files}
train=np.flatnonzero(np.array([r['split']=='train' for r in rows]));probes=train[np.linspace(0,len(train)-1,32,dtype=int)]
with torch.no_grad():actual=model(torch.tensor(features[probes],dtype=torch.float64)).numpy()
expected=np.array([policy.learner.choose(features[i])[1] for i in probes])
joint_error=float(np.abs(actual[:,:6]-expected[:,:6]).max()*model.joint_gain)
grip_error=max(abs(neural_aperture(a)-neural_aperture(b)) for a,b in zip(actual[:,6],expected[:,6]))
if joint_error>.05 or grip_error>.05:raise RuntimeError('Exported head does not match the checkpoint actor')
a.output.mkdir(parents=True)
shutil.copytree(root/'src',a.output/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['head_training_runtime.py','benchmark_training_bundle.py']:shutil.copy2(root/'scripts'/name,a.output/name)
torch.save(model.state_dict(),a.output/'head-state.pt')
orientation_data={}
if a.training_scope!='all':orientation_data['held_rows']=np.array([all(e['observation']['finger_contacts'].values()) for e in examples],dtype=bool)
if a.held_orientation_objective=='cube_up':
    from scipy.spatial.transform import Rotation
    held=np.array([all(e['observation']['finger_contacts'].values()) for e in examples],dtype=bool)
    relative=np.tile(np.array([0.,0.,1.]),(len(examples),1))
    relative_position=np.zeros((len(examples),3),dtype=np.float64);relative_rotation=np.tile(np.eye(3),(len(examples),1,1));cube_sizes=np.full(len(examples),20.,dtype=np.float64)
    task_sizes={}
    for i,e in enumerate(examples):
        if not held[i]:continue
        if e['episode'] not in task_sizes:task_sizes[e['episode']]=json.loads((e['directory']/'manifest.json').read_text())['task']['cube_size_mm']
        cube_sizes[i]=task_sizes[e['episode']]
        observation=e['observation']
        if 'cube_pose' not in observation or 'grasp_pose' not in observation:
            raise ValueError('Tilt supervision requires synchronized teacher pose measurements')
        tool_inverse=Rotation.from_quat(observation['grasp_pose']['quaternion_xyzw']).inv()
        cube_relative=tool_inverse*Rotation.from_quat(observation['cube_pose']['quaternion_xyzw'])
        relative[i]=cube_relative.apply([0,0,1]);relative_rotation[i]=cube_relative.as_matrix()
        relative_position[i]=tool_inverse.apply(np.asarray(observation['cube_pose']['position_mm'])-observation['grasp_pose']['position_mm'])
    orientation_data={'held_rows':held,'relative_cube_up':relative,'relative_cube_position_mm':relative_position,'relative_cube_rotation':relative_rotation,'cube_size_mm':cube_sizes,'current_joints_deg':np.asarray([e['observation']['joints_deg'] for e in examples],dtype=np.float64)}

if a.pregrasp_retention_strength:
    from head_training_runtime import retention_anchor_rows
    if 'held_rows' not in orientation_data:
        orientation_data['held_rows']=np.array([all(e['observation']['finger_contacts'].values()) for e in examples],dtype=bool)
    with torch.no_grad():reference_scores=model(torch.tensor(features,dtype=torch.float64)).numpy()
    anchors=retention_anchor_rows(np.array([r['split']=='train' for r in rows]),orientation_data['held_rows'],np.array([e['recovery_episode'] for e in examples]),a.retention_training_group)
    orientation_data.update(reference_scores=reference_scores,anchor_rows=anchors)
for key in set(targets)&set(orientation_data):
    if not np.array_equal(targets[key],orientation_data[key]):raise ValueError('Conflicting training measurements: '+key)
    orientation_data.pop(key)
np.savez(a.output/'training-data.npz',features=features,**targets,**orientation_data,
         split=np.array([{'train':0,'validation':1,'test':2}[r['split']] for r in rows],dtype=np.int8),
         recovery=np.array([e['recovery_episode'] for e in examples],dtype=bool),
         grasp_rows=np.isin(targets['goal_role_code'],[meta['goal_role_names'].index(role) for role in ['descend','close']]) if 'goal_role_code' in targets else np.array([e['goal_role']=='grasp' for e in examples],dtype=bool))
shutil.copy2(a.checkpoint/'model.npz',a.output/'checkpoint-model.npz')
shutil.copy2(a.checkpoint/'manifest.json',a.output/'checkpoint-manifest.json')
shutil.copy2(a.checkpoint/'parameter-scaling.npy',a.output/'parameter-scaling.npy')
shutil.copy2(a.checkpoint/'feature-rows.json',a.output/'feature-rows.json')
recipe={'orientation_loss_weight':a.orientation_loss_weight,'training_scope':a.training_scope,'precision_validation_weight':a.precision_validation_weight,'gripper_loss_weight':a.gripper_loss_weight,'output_mode':meta['architecture']['output_mode'],'control_hz':meta['architecture']['control_hz'],'parameter_names':[name for name,_ in model.named_parameters()],'motor_shape':list(model.motor_shape),'joint_gain':model.joint_gain,
        'weight_multiple':meta['learning_rule']['weight_multiple'],'motor_weight_multiple':16.,'grasp_goal_height_mm':meta['grasp_goal_height_mm'],
        'grasp_undershoot_penalty':meta['grasp_undershoot_penalty'],'dopamine_multiplier':float(np.tanh(.5)) if meta['dopamine_enabled'] else 0.,
        'validation_aggregation':meta['validation_aggregation'],'checkpoint':str(a.checkpoint.resolve()),'model_sha256':sha256_file(a.checkpoint/'model.npz'),
        'features_sha256':sha256_file(a.checkpoint/'training-features.npz'),'targets_sha256':sha256_file(a.checkpoint/'training-targets.npz'),
        'reconstruction_joint_max_error_deg':joint_error,'reconstruction_gripper_max_error_mm':grip_error,
        'retention_training_group':a.retention_training_group,'retention_selection_group':a.retention_selection_group,'pregrasp_retention_strength':a.pregrasp_retention_strength,'selection_fidelity_limits':{'joint_rmse_deg':a.retention_joint_rmse_limit,'joint_max_abs_deg':a.retention_joint_max_limit,'gripper_aperture_mae_mm':a.retention_aperture_limit} if a.pregrasp_retention_strength else None,
        'held_position_objective':a.held_position_objective,'held_clearance_target_mm':meta.get('desired_lift_clearance_mm',125.),'floor_height_mm':-1.,
        'held_orientation_objective':a.held_orientation_objective,'held_joint_posture_weight':.01 if a.held_orientation_objective=='cube_up' else .04,'held_joint_posture_reference':'current_joints_deg' if a.held_orientation_objective=='cube_up' else 'teacher_joints_deg',
        'orientation_metric_note':'Full tool rotation before grasp; predicted cube up-vector chord error while held' if a.held_orientation_objective=='cube_up' else 'Full tool rotation on all rows',
        'purpose':'Train the same constrained neural head using frozen sensory features; no simulator or teacher in this bundle runtime'}
write_json(a.output/'recipe.json',recipe)
artifact_manifest(a.output,{'schema':'frozen-sensory-head-training-v1','source_checkpoint':str(a.checkpoint.resolve()),'frames':len(rows)})
print(json.dumps({'bundle':str(a.output.resolve()),'bytes':sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file()),'joint_error_deg':joint_error,'gripper_error_mm':grip_error}),flush=True)
