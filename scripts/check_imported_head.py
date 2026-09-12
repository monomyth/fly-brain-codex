"""Compare the exported double-precision training head with the imported actor."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from head_training_runtime import load_bundle
from probability_gripper import neural_aperture
from fly_brain.assets import write_json, sha256_file
from fly_brain.visual_dopamine.motor_policy import MotorPolicy

p=argparse.ArgumentParser(description=__doc__)
for name in ['bundle','result','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
model,_,data,_=load_bundle(a.bundle,'cpu')
with np.load(a.result/'learned-head.npz',allow_pickle=False) as values,torch.no_grad():
    model.weights.copy_(torch.tensor(values['weights']))
    model.excitability.copy_(torch.tensor(values['excitability']))
    model.motor_strength.copy_(torch.tensor(np.abs(values['motor_synapses'])))
with torch.no_grad():expected=model(data['features']).numpy()
policy=MotorPolicy(a.checkpoint,root/'data')
actual=np.array([policy.learner.choose(row)[1] for row in data['features'].numpy()])
joint_error=float(np.max(np.abs(actual[:,:6]-expected[:,:6]))*model.joint_gain)
aperture_error=float(max(abs(neural_aperture(x)-neural_aperture(y)) for x,y in zip(actual[:,6],expected[:,6])))
flips=int(np.count_nonzero((actual[:,6]>=0)!=(expected[:,6]>=0)))
report={'frames':len(actual),'checkpoint':str(a.checkpoint.resolve()),'model_sha256':sha256_file(a.checkpoint/'model.npz'),
        'joint_max_abs_error_deg':joint_error,'continuous_gripper_max_abs_error_mm':aperture_error,
        'binary_gripper_flips':flips,'passed':joint_error<.05 and aperture_error<.05 and flips==0}
write_json(a.output,report);print(json.dumps(report,indent=2),flush=True)
if not report['passed']:raise SystemExit('Imported actor differs from fitted head')
