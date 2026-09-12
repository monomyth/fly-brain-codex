"""Refine only recorded synapses into the seven mapped gripper motor neurons."""
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from scipy.optimize import minimize
from fly_brain.assets import artifact_manifest,write_json
from fly_brain.visual_dopamine.feedback import FeedbackCircuit
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.motor_training import save_checkpoint
from fly_brain.visual_dopamine.trajectory_training import MotorTargetNetwork,load_demonstrations
root=Path('data/benchmark-bundle');source=Path('data/checkpoints/rebot/cuda-hold-20260910-192948');policy=MotorPolicy(source,root)
c=FeedbackCircuit(root/'feedback-circuits'/policy.metadata['feedback_circuit_id'],root)
spec=json.loads(Path('reports/cuda-hold-training-inputs.json').read_text());examples,_=load_demonstrations(spec['datasets'],spec['configurations'],control_hz=2.,grasp_tolerance_mm=12.)
f=np.load(source/'training-features.npz')['features']
assert len(f)==len(examples)
model=MotorTargetNetwork(c,train_motor=True).double().cuda();torch.set_num_threads(2)
with torch.no_grad():
 model.weights.copy_(torch.tensor(policy.learner.weights,device='cuda'));model.excitability.copy_(torch.tensor(policy.learner.excitability,device='cuda'));model.motor_strength.copy_(torch.tensor(np.abs(policy.circuit.motor_from_dn_pm.data),device='cuda'))
model.weights.requires_grad_(False);model.excitability.requires_grad_(False)
rows=c.motor_from_dn_pm.tocoo().row;gripper_rows=np.flatnonzero(c.readout[6]);selected=np.flatnonzero(np.isin(rows,gripper_rows));assert not np.any(c.readout[:6,gripper_rows])
x=torch.tensor(f,dtype=torch.float64,device='cuda');y=torch.tensor([e['target'][6] for e in examples],dtype=torch.float64,device='cuda')
training=[i for i,e in enumerate(examples) if e['split']=='train'];validation=[i for i,e in enumerate(examples) if e['split']=='validation'];assert validation
start=model.motor_strength.detach().cpu().numpy()[selected].copy();bounds=list(zip(np.zeros(len(selected)),16*np.abs(c.motor_from_dn_pm.data[selected])))
with torch.no_grad():joints_before=model(x)[:,:6].clone();motor_before=model.motor_strength.clone()
calls=0;history=[];best=None;best_score=float('inf')
def stats(indices):
 with torch.no_grad():
  scores=model(x[indices])[:,6]*2000
  return {'bce':float(torch.nn.functional.binary_cross_entropy_with_logits(scores,y[indices])),'accuracy':float(((scores>=0)==(y[indices]>.5)).double().mean())}
def objective(v):
 global calls,best,best_score
 with torch.no_grad():model.motor_strength[selected]=torch.tensor(v,device='cuda')
 model.zero_grad();logits=model(x[training])[:,6]*2000
 labels=y[training];weights=torch.where(labels>.5,.5/float(labels.mean()),.5/float(1-labels.mean()))
 loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')*weights).mean()*np.tanh(.5)
 loss.backward();calls+=1
 if calls==1 or calls%50==0:
  score=stats(validation);history.append({'calls':calls,'train':stats(training),'validation':score});print(history[-1],flush=True)
  if score['bce']<best_score:best_score=score['bce'];best=v.copy()
 return float(loss.detach()),model.motor_strength.grad.cpu().numpy()[selected].copy()
res=minimize(objective,start,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':4000,'maxcor':30,'ftol':1e-11,'gtol':1e-7,'maxls':40})
objective(res.x)
if stats(validation)['bce']>best_score:objective(best)
with torch.no_grad():
 unchanged=np.ones(len(motor_before),dtype=bool);unchanged[selected]=False
 assert torch.equal(motor_before[unchanged],model.motor_strength[unchanged])
 joint_difference=float((joints_before-model(x)[:,:6]).abs().max())
 print({'joint_roundoff_max_degrees':joint_difference*1440},flush=True)
 assert joint_difference<1e-10
policy.circuit.motor_from_dn_pm.data=(model.motor_strength*model.motor_signs).detach().cpu().numpy().astype(np.float32)
out=Path('data/checkpoints/rebot')/('cuda-gripper-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
metadata={**policy.metadata,'parent_checkpoint':str(source),'architecture':{**policy.metadata['architecture'],'position_servo':True,'command_scale':.5},'gripper_refinement':{
 'updated_existing_synapses':len(selected),'teacher_label_rule':'Close during final approach within 12 mm and 0.15 rad, or whenever the original teacher requested closure','relabelled_frames':sum(e['gripper_relabelled'] for e in examples),'joint_outputs_unchanged_within_roundoff':True,'joint_roundoff_max_degrees':joint_difference*1440,'train':stats(training),'validation':stats(validation)}}
save_checkpoint(out,policy.core,policy.learner,metadata);write_json(out/'gripper-training.json',history)
m=json.loads((out/'manifest.json').read_text());m.pop('files');artifact_manifest(out,m)
Path('reports/cuda-gripper-model-path.txt').write_text(str(out)+'\n');print({'checkpoint':str(out),'refinement':metadata['gripper_refinement']},flush=True)
