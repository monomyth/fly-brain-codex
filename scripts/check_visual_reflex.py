"""Compare trained double-precision reflex outputs with the live NumPy path."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from fly_brain.assets import write_json,sha256_file
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from probability_gripper import neural_aperture
from visual_reflex_head import load_reflex
from visual_reflex_policy import attach_reflex
p=argparse.ArgumentParser()
for name in ['bundle','inputs','weights','base','output']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[1];torch.set_num_threads(2)
model,fk,data,recipe,origin=load_reflex(a.bundle,a.inputs,'cpu')
with np.load(a.weights,allow_pickle=False) as z,torch.no_grad():model.multipliers.copy_(torch.tensor(z['multipliers'],dtype=torch.float64))
with torch.no_grad():expected=model(data['features']).numpy()
policy=MotorPolicy(a.base,root/'data');attach_reflex(policy,a.inputs,a.weights)
with np.load(a.base/'training-features.npz',allow_pickle=False) as z:old=z['features'].copy()
with np.load(a.inputs/'features.npz',allow_pickle=False) as z:
 extra=z['features'].copy();cpu_rows=z['cpu_rows'];cpu_extra=z['cpu_features']
actual=np.array([policy.learner.choose(np.r_[x,y])[1] for x,y in zip(old,extra)])
joint=float(abs(actual[:,:6]-expected[:,:6]).max()*policy.joint_gain)
grip=max(abs(neural_aperture(x)-neural_aperture(y)) for x,y in zip(actual[:,6],expected[:,6]))
cpu_neural=np.array([policy.learner.choose(np.r_[old[i],value])[1] for i,value in zip(cpu_rows,cpu_extra)])
cpu_joint=float(abs(cpu_neural[:,:6]-actual[cpu_rows,:6]).max()*policy.joint_gain)
cpu_grip=max(abs(neural_aperture(x)-neural_aperture(y)) for x,y in zip(cpu_neural[:,6],actual[cpu_rows,6]))
result={'frames':len(actual),'base':str(a.base.resolve()),'base_model_sha256':sha256_file(a.base/'model.npz'),'base_manifest_sha256':sha256_file(a.base/'manifest.json'),'weights_sha256':sha256_file(a.weights),'reflex_source_sha256':sha256_file(root/'scripts/visual_reflex_policy.py'),'input_manifest_sha256':sha256_file(a.inputs/'manifest.json'),'joint_max_error_deg':joint,'gripper_max_error_mm':grip,'cpu_checked_frames':len(cpu_rows),'cpu_input_joint_max_error_deg':cpu_joint,'cpu_input_gripper_max_error_mm':cpu_grip,'passed':max(joint,cpu_joint)<.05 and max(grip,cpu_grip)<.05}
write_json(a.output,result);print(json.dumps(result,indent=2))
if not result['passed']:raise SystemExit('Reflex runtime parity failed')
