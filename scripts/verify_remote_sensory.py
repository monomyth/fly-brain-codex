"""Validate CUDA/SSH control features against local SciPy without moving the robot."""
import argparse,json,time
from pathlib import Path
import numpy as np
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.remote_features import attach
from fly_brain.visual_dopamine.trajectory_training import load_demonstrations
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--host',default='blacktower.local');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
positions={tuple(json.loads(f.read_text())['task']['cube_xy_mm']) for f in a.dataset.glob('episode-*/manifest.json')}
examples,_=load_demonstrations(a.dataset,{'train':list(positions)});examples=examples[:4]
if not examples:raise ValueError('No complete visual demonstrations')
policy=MotorPolicy(a.checkpoint);t=time.perf_counter();expected=[policy.core.encode_observation(e['observation'],e['directory']).copy() for e in examples];cpu=(time.perf_counter()-t)/len(examples)
attach(policy,a.host)
try:
    elapsed=[];errors=[]
    for e,x in zip(examples,expected):
        t=time.perf_counter();result=policy.core.encode_observation(e['observation'],e['directory']);elapsed.append(time.perf_counter()-t);errors.append(float(np.max(np.abs(result-x))))
    if max(errors)>=1e-4:raise RuntimeError('Remote control features differ from local computation')
    report={'cpu_seconds_per_frame':cpu,'remote_seconds':elapsed,'feature_max_abs_difference':max(errors),'display_neurons':len(policy.core.state),'display_precision_bits':8,'control_features':'float32, separate from lossy display telemetry'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(report)
finally:policy.core.close()
