"""Diagnostic learned robot calibration using frozen MaleCNS motor neuron rates."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from head_training_runtime import load_bundle,loss,statistics

p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=8000);p.add_argument('--input-kind',choices=['motor','sensory'],default='motor');a=p.parse_args()
if a.output.exists():raise ValueError('Use a new experiment directory')
a.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(0)
brain,fk,data,recipe=load_bundle(a.bundle,'cuda')
with torch.no_grad():
 x=data['features'];mbon=torch.tanh(.1*x*brain.input_signs@(brain.weights*brain.mask).T+brain.excitability)
 cb=torch.tanh(torch.sparse.mm(brain.cb_from_mbon,mbon.T).T)
 dn=torch.tanh(torch.sparse.mm(brain.dn_from_mbon_cb,torch.cat([mbon,cb],1).T).T)
 pm=torch.tanh(torch.sparse.mm(brain.pm_from_dn,dn.T).T)
 matrix=torch.sparse_coo_tensor(brain.motor_indices,brain.motor_strength*brain.motor_signs,brain.motor_shape).coalesce()
 motor=torch.tanh(torch.sparse.mm(matrix,torch.cat([dn,pm],1).T).T)
 baseline=(motor@brain.readout.T)*brain.gain-brain.reference
 assert torch.allclose(baseline,brain(x),atol=1e-12,rtol=1e-10)
 train=torch.nonzero(data['split']==0).flatten();validation=torch.nonzero(data['split']==1).flatten()
 signals=motor if a.input_kind=='motor' else x
 mean=signals[train].mean(0);std=signals[train].std(0).clamp_min(1e-10)
 data['features']=torch.cat([(signals-mean)/std,baseline],1).detach()
 # A calibration experiment has its own bound on corrections; no anatomy weights change.
 recipe={**recipe,'pregrasp_retention_strength':0.,'held_orientation_objective':'full_rotation'}

class Decoder(nn.Module):
 def __init__(self):
  super().__init__();self.joint_gain=brain.joint_gain
  self.network=nn.Sequential(nn.Linear(signals.shape[1],128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,7)).double().cuda()
  nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
  self.register_buffer('scale',torch.tensor([20/self.joint_gain]*6+[8/2000],dtype=torch.float64,device='cuda'))
 def forward(self,value):return value[:,-7:]+torch.tanh(self.network(value[:,:-7]))*self.scale

model=Decoder();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
initial=statistics(model,fk,data,validation,recipe);best=float('inf');best_state=None;history=[];start=time.perf_counter()
for step in range(a.steps):
 selected=train[torch.randint(len(train),(min(512,len(train)),),device='cuda')]
 optimizer.zero_grad();objective=loss(model,fk,data,selected,recipe);objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
 if (step+1)%200==0:
  measured=statistics(model,fk,data,validation,recipe)
  score=measured['goal_position_rmse_mm']+2*measured['goal_orientation_rmse_deg']+30*(1-measured['gripper_accuracy'])
  if score<best:best=score;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};best_step=step+1
  row={'step':step+1,'elapsed_seconds':time.perf_counter()-start,'selection':measured,'best_step':best_step};history.append(row)
  (a.output/'progress.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
model.load_state_dict(best_state)
result={'scope':'Diagnostic nonlinear motor-neuron-to-robot calibration; no native qualification or UI deployment',
 'inputs':('Frozen MaleCNS motor neuron responses' if a.input_kind=='motor' else 'Sensory neuron responses before the motor path (diagnostic bypass)')+' and existing fixed readout; no pixels, cube labels or task state at inference',
 'input_kind':a.input_kind,'decoder_inputs':signals.shape[1],'motor_neurons':motor.shape[1],'frames':len(x),'initial_validation':initial,'validation':statistics(model,fk,data,validation,recipe),'best_step':best_step,'elapsed_seconds':time.perf_counter()-start,'source_model_sha256':recipe['model_sha256'],'brain_weights_unchanged':True,'device':torch.cuda.get_device_name()}
np.savez(a.output/'decoder.npz',**{k:v.cpu().numpy() for k,v in model.state_dict().items()},mean=mean.cpu().numpy(),std=std.cpu().numpy())
(a.output/'result.json').write_text(json.dumps(result,indent=2));(a.output/'history.json').write_text(json.dumps(history));print(json.dumps(result),flush=True)
