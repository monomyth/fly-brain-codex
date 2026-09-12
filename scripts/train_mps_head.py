"""Fit the unchanged anatomical control head locally on Apple Metal."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from fly_brain.assets import write_json,sha256_file,artifact_manifest
from fly_brain.visual_dopamine.feedback_training import validation_score
from head_training_runtime import load_bundle,loss,statistics
from mps_head import DenseMotorHead,mps_data

p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=16000);a=p.parse_args()
if a.output.exists() or not torch.backends.mps.is_available():raise ValueError('Use a new output directory and an available Metal device')
a.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(0)
source,cpu_fk,cpu_data,recipe=load_bundle(a.bundle,'cpu');model=DenseMotorHead(source).to('mps');data=mps_data(cpu_data);fk=type(cpu_fk)().float().to('mps')
train=torch.nonzero(data['split']==0).flatten();validation=torch.nonzero(data['split']==1).flatten();indices=source.motor_indices.cpu().numpy()
probes=train[:128];cpu_rows=probes.cpu()
source.zero_grad();cpu_value=loss(source,cpu_fk,cpu_data,cpu_rows,recipe);cpu_value.backward()
model.zero_grad();gpu_value=loss(model,fk,data,probes,recipe);gpu_value.backward();torch.mps.synchronize()
cpu_grad=torch.cat([source.weights.grad.flatten(),source.excitability.grad,source.motor_strength.grad]).detach().numpy()
gpu_grad=np.r_[model.weights.grad.cpu().numpy().ravel(),model.excitability.grad.cpu().numpy(),model.motor_strength.grad.cpu().numpy()[indices[0],indices[1]]]
relative=float(np.linalg.norm(gpu_grad-cpu_grad)/max(np.linalg.norm(cpu_grad),1e-12));loss_error=abs(float(gpu_value.detach())-float(cpu_value.detach()))/max(abs(float(cpu_value.detach())),1e-12)
if relative>.005 or loss_error>.0005:raise RuntimeError('Metal training gradient or loss differs from CPU reference')
fixed={name:value.detach().cpu().clone() for name,value in model.named_buffers()}

def parameters():return {name:value.detach().cpu().clone() for name,value in model.named_parameters()}
def restore(values):
 with torch.no_grad():
  for name,value in model.named_parameters():value.copy_(values[name].to('mps'))
def selection():
 result=statistics(model,fk,data,validation,recipe)
 result['groups']={name:statistics(model,fk,data,validation[data['recovery'][validation]==value],recipe) for name,value in [('legacy',False),('recovery',True)]}
 return result
initial_parameters=parameters();initial=selection();best_score=validation_score(initial,True);best=initial_parameters;best_step=0
# Set a conservative initial Adam step using training-only measured command change.
optimizer=torch.optim.Adam(model.parameters(),lr=.001)
with torch.no_grad():before=model(data['features'][probes]).detach().clone()
model.zero_grad();loss(model,fk,data,probes,recipe).backward();optimizer.step();model.project(recipe['weight_multiple'])
with torch.no_grad():change=float(((model(data['features'][probes])[:,:6]-before[:,:6])*model.joint_gain).square().mean().sqrt())
learning_rate=float(np.clip(.001*.1/max(change,1e-9),1e-7,1e-3));restore(initial_parameters)
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate);rng=np.random.default_rng(0);history=[];start=time.perf_counter()
write_json(a.output/'benchmark.json',{'device':'Apple Metal (MPS)','loss_relative_error':loss_error,'gradient_relative_l2_error':relative,'learning_rate':learning_rate,'initial_step_joint_rms_deg':change,'passed':True})
for step in range(a.steps):
 sampled=train[torch.tensor(rng.integers(len(train),size=256),device='mps')]
 optimizer.zero_grad();objective=loss(model,fk,data,sampled,recipe);objective.backward();optimizer.step();model.project(recipe['weight_multiple'])
 if (step+1)%200==0:
  stats=selection();score=validation_score(stats,True);limits=recipe.get('selection_fidelity_limits');fidelity=stats['groups']['legacy'].get('nonheld_fidelity',{})
  eligible=not limits or all(fidelity.get(k,float('inf'))<=v for k,v in limits.items())
  if eligible and score<best_score:best_score=score;best=parameters();best_step=step+1
  record={'step':step+1,'seconds':time.perf_counter()-start,'loss':float(objective.detach()),'selection':stats,'best_step':best_step,'eligible':eligible};history.append(record);write_json(a.output/'progress.json',record);print(json.dumps(record),flush=True)
restore(best)
for name,value in model.named_buffers():
 if not torch.equal(value.cpu(),fixed[name]):raise AssertionError('Fixed anatomical buffer changed')
values=parameters();weights=values['weights'].numpy();offset=values['excitability'].numpy();strength=values['motor_strength'].numpy()[indices[0],indices[1]]
synapses=strength*source.motor_signs.detach().numpy()
assert np.all(weights>=0) and np.all(weights<=recipe['weight_multiple']*source.base.detach().numpy()+1e-6)
assert np.all(strength>=0) and np.all(strength<=16*source.motor_base.detach().numpy()+1e-6)
np.savez(a.output/'learned-head.npz',weights=weights,excitability=offset,motor_synapses=synapses)
metrics={name:statistics(model,fk,data,torch.nonzero(data['split']==i).flatten(),recipe) for i,name in enumerate(['train','validation','test'])}
result={'source_model_sha256':recipe['model_sha256'],'source_bundle_manifest_sha256':sha256_file(a.bundle/'manifest.json'),'trainer_sha256':sha256_file(__file__),'training_device':'mps','optimizer':'Adam with anatomical projection','iterations_requested':a.steps,'optimizer_iterations':a.steps,'optimizer_message':'Completed local Metal training; selected validation-best eligible step','elapsed_seconds':time.perf_counter()-start,'initial_selection':initial,'selection_statistics':selection(),'best_step':best_step,'selection':'Balanced validation groups, with declared legacy-fidelity limits','metrics':metrics,'fixed_buffers_unchanged':True,'parameter_bounds_passed':True,'loss_recipe':recipe,'closed_loop_validated':False}
write_json(a.output/'result.json',result);write_json(a.output/'training-history.json',history);artifact_manifest(a.output,{'schema':'local-metal-head-fit-v1'});print(json.dumps(result,indent=2),flush=True)
