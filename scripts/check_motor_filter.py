"""Replay deterministic actuator filtering on CPU-reference and MLX features."""
import argparse,json
from pathlib import Path
import numpy as np
from fly_brain.assets import sha256_file,write_json
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.adapters import JOINT_LOWER,JOINT_UPPER
from probability_gripper import neural_aperture
from motor_output_filter import MotorOutputFilter,DEFAULTS
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--parity',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];cp=a.checkpoint.resolve();proof=json.loads(a.parity.read_text());policy=MotorPolicy(cp,root/'data')
checks={'model_sha256':sha256_file(cp/'model.npz'),'manifest_sha256':sha256_file(cp/'manifest.json'),'backend_source_sha256':sha256_file(root/'scripts/mlx_sensory_backend.py'),'gripper_response_source_sha256':sha256_file(root/'scripts/probability_gripper.py')}
if not proof['passed'] or any(proof.get(k)!=v for k,v in checks.items()):raise ValueError('Matching complete raw parity is required')
rows=json.loads((cp/'feature-rows.json').read_text());saved=Path(proof['saved_sensory_features'])
if sha256_file(saved)!=proof['saved_sensory_features_sha256']:raise ValueError('Cached MLX features changed')
with np.load(cp/'training-features.npz',allow_pickle=False) as z:reference=z['features']
with np.load(saved,allow_pickle=False) as z:
 actual=z['features'];assert np.array_equal(z['row_indices'],np.arange(len(rows)))
assert len(rows)==len(reference)==len(actual)==proof['frames']
folders={f.name:f for d in policy.metadata['datasets'] for f in Path(d).glob('episode-*')};last=None;cache={};stages=[MotorOutputFilter(),MotorOutputFilter()];joints=0.;grip=0.;flips=0
for i,row in enumerate(rows):
 ep=row['episode']
 if ep!=last:
  for s in stages:s.reset()
  last=ep
 if ep not in cache:cache[ep]=[json.loads(line) for line in (folders[ep]/'steps.jsonl').read_text().splitlines()]
 obs=cache[ep][row['step']]['observation'];actions=[]
 for features,stage in zip((reference[i],actual[i]),stages):
  scores=policy.learner.choose(features)[1]
  action={'joints_deg':np.clip(scores[:6]*policy.joint_gain,JOINT_LOWER,JOINT_UPPER).tolist(),'gripper_mm':neural_aperture(scores[6])}
  actions.append(stage.apply(action,obs['simulation_time']))
 joints=max(joints,float(np.abs(np.asarray(actions[0]['joints_deg'])-actions[1]['joints_deg']).max()));grip=max(grip,abs(actions[0]['gripper_mm']-actions[1]['gripper_mm']));flips+=int(stages[0].closed!=stages[1].closed)
write_json(a.output,{'checkpoint':str(cp),'parameters':DEFAULTS,'frames':len(rows),'raw_parity':str(a.parity.resolve()),'raw_parity_sha256':sha256_file(a.parity),'filter_source_sha256':sha256_file(root/'scripts/motor_output_filter.py'),'joint_max_abs_error_deg':joints,'gripper_max_abs_error_mm':grip,'state_mismatches':flips,'passed':joints<.05 and grip<.05 and flips==0,**checks})
print(a.output.read_text())
