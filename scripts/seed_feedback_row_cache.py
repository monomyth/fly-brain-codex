"""Validate and index a checkpoint's existing sensory cache without recomputation."""
import argparse,json
from pathlib import Path
import numpy as np
from fly_brain.assets import canonical_hash
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.trajectory_training import load_demonstrations
from fly_brain.visual_dopamine.feedback_cache import calibration_recipe,cached_responses

parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True);args=parser.parse_args()
root=Path(__file__).resolve().parents[1];run=root/'reports/two-view-20260910-233652'
p=MotorPolicy(args.checkpoint,root/'data');m=p.metadata
examples,_=load_demonstrations(m['datasets'],json.loads((run/'training-configurations.json').read_text()),control_hz=2,goal_supervision=m['goal_supervision'],sample_hz=m['sample_hz'],grasp_goal_height_mm=m.get('grasp_label_height_mm'))
retinal=np.asarray([p.core.encoder.sample(e['observation']['images'],e['directory']) for e in examples])
body=np.asarray([p.core.body_encoder.sample(e['observation']) for e in examples]);train=[i for i,e in enumerate(examples) if e['split']=='train']
recipe=calibration_recipe(p.core,m['training_device'])
key=canonical_hash({'schema':2,**recipe,'retinal':canonical_hash(retinal.tolist()),'body':canonical_hash(body.tolist()),'train_indices':train})
if key!=m['sensory_feature_cache_key']:raise ValueError('Legacy cache does not match the current inputs and calibration')
with np.load(root/'data/feature-cache/feedback'/f'{key}.npz',allow_pickle=False) as data:raw=data['raw_responses'].copy()
path=root/'data/feature-cache/feedback/rows'/f"{canonical_hash({'schema':1,**recipe})}.npz"
def unexpected(*args):raise AssertionError('Verified legacy cache should cover every row')
result,stats=cached_responses(path,retinal,body,len(p.circuit.kc),unexpected,raw)
assert np.array_equal(result,raw)
print({'cache':str(path),'legacy_cache_verified':True,'bitwise_response_match':True,**stats})
