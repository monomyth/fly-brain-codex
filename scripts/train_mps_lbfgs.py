"""Local Metal continuation with the established scaled L-BFGS-B objective."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from fly_brain.assets import write_json,sha256_file,artifact_manifest
from fly_brain.visual_dopamine.feedback_training import validation_score
from head_training_runtime import load_bundle,loss,statistics
from mps_head import DenseMotorHead,mps_data

p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=6000);a=p.parse_args()
if a.output.exists() or not torch.backends.mps.is_available():raise ValueError('Use a new output and available Metal device')
a.output.mkdir(parents=True);torch.set_num_threads(2)
source,cpu_fk,cpu_data,recipe=load_bundle(a.bundle,'cpu');model=DenseMotorHead(source).to('mps');data=mps_data(cpu_data);fk=type(cpu_fk)().float().to('mps')
post,pre=torch.nonzero(source.mask,as_tuple=True);post=post.to('mps');pre=pre.to('mps');mi=source.motor_indices.to('mps');n=len(post);mb=model.excitability.numel()
scale=np.load(a.bundle/'parameter-scaling.npy',allow_pickle=False)

def vector(gradient=False):
 def value(p):return p.grad if gradient else p.detach()
 return torch.cat([value(model.weights)[post,pre],value(model.excitability),value(model.motor_strength)[mi[0],mi[1]]]).detach().cpu().numpy().astype(np.float64)
def assign(value):
 value=torch.tensor(value,dtype=torch.float32,device='mps')
 with torch.no_grad():
  model.weights.zero_();model.weights[post,pre]=value[:n];model.excitability.copy_(value[n:n+mb]);model.motor_strength.zero_();model.motor_strength[mi[0],mi[1]]=value[n+mb:]
start_vector=vector();low=np.r_[np.zeros(n),np.full(mb,-2.),np.zeros(len(start_vector)-n-mb)]
high=np.r_[recipe['weight_multiple']*source.base[source.mask].cpu().numpy(),np.full(mb,2.),16*source.motor_base.cpu().numpy()]
if scale.shape!=start_vector.shape:raise ValueError('Preconditioner shape mismatch')
scope=data['held_rows'] if recipe.get('training_scope')=='held' else ~data['held_rows'] if recipe.get('training_scope')=='unheld' else torch.ones_like(data['split'],dtype=torch.bool)
train=torch.nonzero((data['split']==0)&scope).flatten();validation=torch.nonzero((data['split']==1)&scope).flatten()
def selection():
 s=statistics(model,fk,data,validation,recipe);s['groups']={name:statistics(model,fk,data,validation[data['recovery'][validation]==v],recipe) for name,v in [('legacy',False),('recovery',True)]};return s
def selection_score(stats):
 if stats.get('groups'):return float(np.mean([selection_score(s) for s in stats['groups'].values()]))
 score=validation_score(stats,True);critical=stats.get('grasp_precision')
 if critical:
  score+=recipe.get('precision_validation_weight',0.)*(critical['xy_rmse_mm']/3+critical['height_rmse_mm']/3+critical['orientation_rmse_deg']/5+3*(1-critical['gripper_accuracy']))
 return score
initial=selection();best_score=selection_score(initial);best=start_vector.copy();best_call=0;calls=0;history=[];start=time.perf_counter()
fixed={name:value.detach().cpu().clone() for name,value in model.named_buffers()}
def objective(scaled):
 global calls,best,best_score,best_call
 value=scaled/scale;assign(value);model.zero_grad();objective=loss(model,fk,data,train,recipe);objective.backward();gradient=vector(True);measured=float(objective.detach());calls+=1
 if not np.isfinite(measured) or not np.isfinite(gradient).all():raise FloatingPointError('Nonfinite Metal objective')
 if calls==1 or calls%100==0:
  stats=selection();score=selection_score(stats);fidelity=(stats['groups']['legacy'] if recipe.get('retention_selection_group','all')=='legacy' else stats).get('nonheld_fidelity',{});limits=recipe.get('selection_fidelity_limits');eligible=not limits or all(fidelity.get(k,float('inf'))<=v for k,v in limits.items())
  if eligible and score<best_score:best=value.copy();best_score=score;best_call=calls
  record={'calls':calls,'seconds':time.perf_counter()-start,'loss':measured,'selection':stats,'best_call':best_call,'eligible':eligible};history.append(record);write_json(a.output/'progress.json',record);print(json.dumps(record),flush=True)
 return measured,gradient/scale
result=minimize(objective,start_vector*scale,jac=True,bounds=list(zip(low*scale,high*scale)),method='L-BFGS-B',options={'maxiter':a.iterations,'maxfun':a.iterations*3,'maxls':40,'maxcor':50,'ftol':1e-7,'gtol':1e-5})
assign(best)
for name,value in model.named_buffers():
 if not torch.equal(value.cpu(),fixed[name]):raise AssertionError('Fixed anatomical buffer changed')
weights=model.weights.detach().cpu().numpy();offset=model.excitability.detach().cpu().numpy();strength=model.motor_strength.detach()[mi[0],mi[1]].cpu().numpy();synapses=strength*source.motor_signs.cpu().numpy()
assert np.all(weights>=0) and np.all(weights<=recipe['weight_multiple']*source.base.cpu().numpy()+1e-6)
assert np.all(strength>=0) and np.all(strength<=16*source.motor_base.cpu().numpy()+1e-6)
np.savez(a.output/'learned-head.npz',weights=weights,excitability=offset,motor_synapses=synapses)
metrics={name:statistics(model,fk,data,torch.nonzero((data['split']==i)&scope).flatten(),recipe) for i,name in enumerate(['train','validation','test'])}
report={'source_model_sha256':recipe['model_sha256'],'source_bundle_manifest_sha256':sha256_file(a.bundle/'manifest.json'),'trainer_sha256':sha256_file(__file__),'training_device':'mps','optimizer':'RMS-scaled L-BFGS-B in float32 on Metal','iterations_requested':a.iterations,'optimizer_iterations':int(result.nit),'optimizer_message':str(result.message),'elapsed_seconds':time.perf_counter()-start,'initial_selection':initial,'selection_statistics':selection(),'best_call':best_call,'selection':'Balanced validation groups with the declared grasp-precision weight and any legacy-fidelity limits','metrics':metrics,'fixed_buffers_unchanged':True,'parameter_bounds_passed':True,'loss_recipe':recipe,'closed_loop_validated':False}
write_json(a.output/'result.json',report);write_json(a.output/'training-history.json',history);artifact_manifest(a.output,{'schema':'local-metal-head-fit-v1'});print(json.dumps(report,indent=2),flush=True)
