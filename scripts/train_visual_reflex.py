"""Fit only recorded visual-to-descending connections; the old pathway stays fixed."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from head_training_runtime import loss,statistics
from visual_reflex_head import load_reflex
from fly_brain.visual_dopamine.feedback_training import validation_score
from fly_brain.assets import write_json,sha256_file

p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--reflex',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=5000);a=p.parse_args()
if a.output.exists():raise ValueError('Use a new result directory')
a.output.mkdir(parents=True);torch.set_num_threads(2);model,fk,data,recipe,origin=load_reflex(a.bundle,a.reflex,'cuda')
recipe={**recipe,'pregrasp_retention_strength':50.}
train=torch.nonzero(data['split']==0).flatten();validation=torch.nonzero(data['split']==1).flatten()

def selection():
    r=statistics(model,fk,data,validation,recipe)
    r['groups']={name:statistics(model,fk,data,validation[data['recovery'][validation]==value],recipe) for name,value in [('legacy',False),('recovery',True)]}
    return r
initial=selection();best_score=validation_score(initial,True);best=np.zeros(model.multipliers.numel());best_call=0;calls=0;history=[];start=time.perf_counter()
# RMS neuronal-output Jacobian scaling on training probes only.
probes=train[torch.linspace(0,len(train)-1,16,device='cuda').long()]
predicted=model(data['features'][probes]);sum_square=torch.zeros_like(model.multipliers)
for row in range(len(probes)):
 for axis in range(7):
  grad=torch.autograd.grad(predicted[row,axis]*(model.joint_gain if axis<6 else 2000),model.multipliers,retain_graph=True)[0]
  sum_square+=grad.square()
scale=(sum_square/(len(probes)*7)).sqrt().detach().cpu().numpy();positive=scale[scale>0];scale=np.maximum(scale,max(float(np.median(positive))*.05,1e-10));np.save(a.output/'parameter-scaling.npy',scale)

def assign(value):
 with torch.no_grad():model.multipliers.copy_(torch.tensor(value,dtype=torch.float64,device='cuda'))
def objective(scaled):
 global calls,best,best_score,best_call
 value=scaled/scale;assign(value);model.zero_grad();objective=loss(model,fk,data,train,recipe)
 objective.backward();gradient=model.multipliers.grad.detach().cpu().numpy().copy();measured=float(objective.detach());calls+=1
 if not np.isfinite(measured) or not np.isfinite(gradient).all():raise FloatingPointError('Nonfinite reflex loss')
 if calls==1 or calls%100==0:
  stats=selection();score=validation_score(stats,True);fidelity=stats['groups']['legacy']['nonheld_fidelity']
  eligible=fidelity['joint_rmse_deg']<=.25 and fidelity['joint_max_abs_deg']<=1.25 and fidelity['gripper_aperture_mae_mm']<=1.
  if eligible and score<best_score:best=value.copy();best_score=score;best_call=calls
  record={'calls':calls,'seconds':time.perf_counter()-start,'loss':measured,'selection':stats,'best_call':best_call,'eligible':eligible};history.append(record);write_json(a.output/'progress.json',record);print(json.dumps(record),flush=True)
 return measured,gradient/scale
result=minimize(objective,np.zeros_like(scale),jac=True,bounds=list(zip(np.zeros_like(scale),64*scale)),method='L-BFGS-B',options={'maxiter':a.iterations,'maxfun':a.iterations*3,'maxls':40,'maxcor':30,'ftol':1e-11,'gtol':1e-7})
assign(best)
np.savez(a.output/'reflex-model.npz',multipliers=best.astype(np.float32))
report={'kind':'measured_visual_reflex_synapses','source_model_sha256':recipe['model_sha256'],'reflex_manifest_sha256':sha256_file(a.reflex/'manifest.json'),'trainable_synapses':len(best),'new_connections':0,'old_pathway_frozen':True,'initial_selection':initial,'selection':selection(),'best_call':best_call,'iterations':int(result.nit),'calls':calls,'seconds':time.perf_counter()-start,'bounds_passed':bool(np.all((best>=0)&(best<=64))),'device':torch.cuda.get_device_name(),'closed_loop_validated':False}
write_json(a.output/'result.json',report);write_json(a.output/'history.json',history);print(json.dumps(report,indent=2),flush=True)
