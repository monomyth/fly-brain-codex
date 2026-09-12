"""Qualify the measured visual-reflex branch in native simulation."""
import argparse,json,fcntl
from pathlib import Path
from fly_brain.assets import sha256_file,write_json
from fly_brain.simulator import EpisodeSimulator
from fly_brain.visual_dopamine.motor_run import run
from probability_gripper import ProbabilityGripPolicy
from mlx_sensory_backend import MLXFeedbackCore
from visual_reflex_policy import attach_reflex
p=argparse.ArgumentParser()
for name in ['base','inputs','weights','parity','tasks','output']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[1]
lock=(root/'.runtime/native-qualification.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
proof=json.loads(a.parity.read_text());checks={'base_model_sha256':sha256_file(a.base/'model.npz'),'base_manifest_sha256':sha256_file(a.base/'manifest.json'),'weights_sha256':sha256_file(a.weights),'reflex_source_sha256':sha256_file(root/'scripts/visual_reflex_policy.py'),'input_manifest_sha256':sha256_file(a.inputs/'manifest.json')}
if not proof['passed'] or any(proof.get(k)!=v for k,v in checks.items()):raise ValueError('Matching reflex numerical proof required')
if a.output.exists():raise ValueError('Use a new output directory')
a.output.mkdir(parents=True);policy=ProbabilityGripPolicy(a.base,root/'data');policy.core=MLXFeedbackCore(policy.core);attach_reflex(policy,a.inputs,a.weights)
write_json(a.output/'runtime.json',{'kind':'visual_reflex','backend':'MLX Metal','gripper_response':'probability','control_hz':2.,'max_steps':160,'settle_seconds':.1,'learn':False,'reflex':policy.reflex_runtime,**checks})
with EpisodeSimulator(log=a.output/'native.log') as client:report=run(client,policy,json.loads(a.tasks.read_text()),a.output,max_steps=160,learn=False,settle_seconds=.1,stop_after_failures=3)
report['controller_kind']='visual_reflex';report['reflex_runtime']=policy.reflex_runtime;write_json(a.output/'evaluation.json',report)
print(json.dumps({'successes':report['successes'],'attempts':report['attempts']}),flush=True)
