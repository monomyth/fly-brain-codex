"""Compare actual MaleCNS sensory and motor operations on CPU and CUDA."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.feedback import TorchFeedbackEncoder
from fly_brain.visual_dopamine.trajectory_training import MotorTargetNetwork
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--samples',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(0)
policy=MotorPolicy(a.bundle/'checkpoint',a.bundle);core=policy.core;data=np.load(a.samples)
r=data['retinal'];b=data['body'];report={'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'free_total_bytes':torch.cuda.mem_get_info()}
t=time.perf_counter();cpu=np.asarray([core.encode_responses(core.response(x,y)) for x,y in zip(r[:2],b[:2])]);report['scipy_feature_seconds_per_frame']=(time.perf_counter()-t)/2
encoder=TorchFeedbackEncoder(core,'cuda');encoder.responses(r[:2],b[:2]);torch.cuda.synchronize()
t=time.perf_counter();gpu=core.encode_responses(encoder.responses(r,b));torch.cuda.synchronize();report['cuda_feature_seconds_per_frame']=(time.perf_counter()-t)/len(r)
report['feature_max_abs_difference']=float(np.max(np.abs(cpu-gpu[:2])))
report['motor_iterations']={}
for device in ['cpu','cuda']:
    model=MotorTargetNetwork(policy.circuit).double().to(device)
    features=torch.tensor(np.tile(gpu,(8,1)),dtype=torch.float64,device=device)
    for _ in range(2):model.zero_grad();model(features).square().mean().backward()
    if device=='cuda':torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(20):model.zero_grad();model(features).square().mean().backward()
    if device=='cuda':torch.cuda.synchronize()
    report['motor_iterations'][device]=(time.perf_counter()-t)/20
report['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
