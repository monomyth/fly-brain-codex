"""Compare each tactile-selected action with its original frozen response bank."""
import argparse,json
from pathlib import Path
import numpy as np
from fly_brain.assets import sha256_file,write_json
from probability_gripper import ProbabilityGripPolicy

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[1]
combined=ProbabilityGripPolicy(a.checkpoint,root/'data');config=combined.metadata['touch_gated_hold']
originals={name:ProbabilityGripPolicy(config[name+'_checkpoint'],root/'data') for name in ('pickup','holding')}
for name,policy in originals.items():
    if sha256_file(policy.checkpoint/'model.npz')!=config[name+'_model_sha256']:raise ValueError('An original response bank changed')
class ReplayCore:
    def __init__(self,state):self.state=np.zeros_like(state);self.features=None
    def reset(self):self.state.fill(0)
    def encode_observation(self,*args,**kwargs):return self.features
for policy in [combined,*originals.values()]:policy.core=ReplayCore(policy.core.state)
rows=json.loads((a.checkpoint/'feature-rows.json').read_text())
with np.load(a.checkpoint/'training-features.npz') as z:features=z['features'].copy()
folders={folder.name:folder for d in combined.metadata['datasets'] for folder in Path(d).glob('episode-*')};cache={};counts={'pickup':0,'holding':0};maximum=0.
for row,value in zip(rows,features):
    ep=row['episode']
    if ep not in cache:cache[ep]=[json.loads(line) for line in (folders[ep]/'steps.jsonl').read_text().splitlines()]
    observation=cache[ep][row['step']]['observation']
    # Only explicitly allowed body/timing fields reach the action decoder.
    obs={k:observation[k] for k in ('episode_id','frame_id','simulation_time','joints_deg','gripper_mm','finger_contacts','images')}
    combined.core.features=value;combined.reset(obs);actual=combined.act(obs)
    mode=combined.active_motor_bank;source=originals[mode];source.core.features=value;source.reset(obs);expected=source.act(obs)
    difference=np.r_[np.asarray(actual['joints_deg'])-expected['joints_deg'],actual['gripper_mm']-expected['gripper_mm']]
    maximum=max(maximum,float(abs(difference).max()));counts[mode]+=1
report={'checkpoint':str(a.checkpoint.resolve()),'model_sha256':sha256_file(a.checkpoint/'model.npz'),
        'manifest_sha256':sha256_file(a.checkpoint/'manifest.json'),'holding_model_sha256':sha256_file(a.checkpoint/'holding-model.npz'),
        'motor_policy_source_sha256':sha256_file(root/'src/fly_brain/visual_dopamine/motor_policy.py'),
        'verification_source_sha256':sha256_file(__file__),'frames':len(rows),'bank_frames':counts,
        'action_max_abs_error':maximum,'passed':maximum==0 and min(counts.values())>0,
        'scope':'Every recorded body state selects a bank; all seven decoded commands must exactly match that original frozen policy.'}
write_json(a.output,report);print(json.dumps(report,indent=2))
if not report['passed']:raise SystemExit('Touch selection changed an original response')
