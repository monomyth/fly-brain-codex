"""Run a frozen checkpoint only after complete offline MLX equivalence validation."""
import argparse
import math
import json
from pathlib import Path
from fly_brain.assets import sha256_file, write_json
from fly_brain.simulator import EpisodeSimulator
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.motor_run import run
from mlx_sensory_backend import MLXFeedbackCore

p=argparse.ArgumentParser(description=__doc__)
for name in ['checkpoint','tasks','parity','output']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--stop-after-failures',type=int)
p.add_argument('--motor-filter-parity',type=Path)
p.add_argument('--control-hz',type=float)
p.add_argument('--cadence-parity',type=Path)
p.add_argument('--max-steps',type=int,default=160)
p.add_argument('--settle-seconds',type=float,default=.1)
p.add_argument('--gripper-response',choices=['binary','probability'],default='binary')
a=p.parse_args();root=Path(__file__).resolve().parents[1]
# Serialize native qualification runs on this Mac; GPU fitting may run remotely in parallel.
import fcntl
lock_directory=root/'.runtime';lock_directory.mkdir(exist_ok=True)
native_lock=(lock_directory/'native-qualification.lock').open('a')
fcntl.flock(native_lock.fileno(),fcntl.LOCK_EX)
parity=json.loads(a.parity.read_text());rows=json.loads((a.checkpoint/'feature-rows.json').read_text())
checks={'model_sha256':sha256_file(a.checkpoint/'model.npz'),'manifest_sha256':sha256_file(a.checkpoint/'manifest.json'),
        'backend_source_sha256':sha256_file(Path(__file__).with_name('mlx_sensory_backend.py'))}
if a.gripper_response=='probability':checks['gripper_response_source_sha256']=sha256_file(Path(__file__).with_name('probability_gripper.py'))
if (not parity['passed'] or parity['checkpoint']!=str(a.checkpoint.resolve()) or parity['frames']!=len(rows)
        or parity['episodes']!=len({r['episode'] for r in rows}) or any(parity.get(k)!=v for k,v in checks.items())):
    raise ValueError('Complete parity results must match this exact checkpoint and backend')
if a.gripper_response=='probability' and not parity.get('continuous_gripper_max_abs_error_mm',float('inf'))<.05:raise ValueError('Continuous aperture parity is required')
from touch_gate_validation import validate_touch_gate
validate_touch_gate(a.checkpoint,parity)
if a.control_hz is not None:
    if not math.isfinite(a.control_hz) or not 1<=a.control_hz<=10 or a.cadence_parity is None:raise ValueError('A 1-10 Hz override requires cadence parity')
    cadence=json.loads(a.cadence_parity.read_text())
    if not cadence['passed'] or abs(cadence['interval_seconds']-1/a.control_hz)>1e-9 or any(cadence.get(k)!=v for k,v in checks.items()):raise ValueError('Cadence parity does not match this runtime')
filter_proof=None
if a.motor_filter_parity:
    from motor_output_filter import attach_filter,DEFAULTS
    filter_proof=json.loads(a.motor_filter_parity.read_text())
    if (a.gripper_response!='probability' or not filter_proof['passed'] or filter_proof['parameters']!=DEFAULTS or filter_proof['frames']!=len(rows) or filter_proof['filter_source_sha256']!=sha256_file(root/'scripts/motor_output_filter.py') or any(filter_proof.get(k)!=v for k,v in checks.items()) or filter_proof['raw_parity_sha256']!=sha256_file(a.parity)):
        raise ValueError('Matching full actuator-filter parity is required')
if a.output.exists() or not a.output.resolve().is_relative_to(root):raise ValueError('Use a new project-local output directory')
a.output.mkdir(parents=True)
write_json(a.output/'runtime.json',{'motor_filter':filter_proof,'runtime_source_sha256':sha256_file(root/'src/fly_brain/visual_dopamine/motor_run.py'),'runner_source_sha256':sha256_file(__file__),'backend':'MLX Metal','gripper_response':a.gripper_response,'parity':str(a.parity.resolve()),'learn':False,'control_hz_override':a.control_hz,'cadence_parity':str(a.cadence_parity) if a.cadence_parity else None,'max_steps':a.max_steps,'settle_seconds':a.settle_seconds,**checks})
policy_type=MotorPolicy
if a.gripper_response=='probability':
    from probability_gripper import ProbabilityGripPolicy
    policy_type=ProbabilityGripPolicy
policy=policy_type(a.checkpoint,root/'data');policy.core=MLXFeedbackCore(policy.core)
if filter_proof:attach_filter(policy)
if a.control_hz is not None:
    if policy.output_mode!='joint_targets':raise ValueError('Cadence experiment requires absolute joint targets')
    policy.hz=a.control_hz
with EpisodeSimulator(log=a.output/'native.log') as client:
    report=run(client,policy,json.loads(a.tasks.read_text()),a.output,max_steps=a.max_steps,learn=False,settle_seconds=a.settle_seconds,stop_after_failures=a.stop_after_failures)
if sha256_file(a.checkpoint/'model.npz')!=checks['model_sha256']:raise RuntimeError('Model weights changed during evaluation')
print(json.dumps({'attempts':report['attempts'],'successes':report['successes'],'stopped_early':bool(report.get('stopped_early_after_failures')),'execution_blocked':report.get('execution_blocked'),'startup_failures':report.get('startup_failures',0)}),flush=True)
