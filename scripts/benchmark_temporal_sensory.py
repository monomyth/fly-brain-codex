"""Compare sequence batching with the actual live temporal encoder on recorded RGB."""
import json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from fly_brain.assets import write_json
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.trajectory_training import load_demonstrations
from fly_brain.visual_dopamine.temporal import TemporalFeedbackCore,TemporalFeedbackEncoder

root=Path(__file__).resolve().parents[1];run=root/'reports/temporal-control-20260911'
p=MotorPolicy(root/'data/checkpoints/rebot/two-view-v4-common125-cpu-seed0-20260911',root/'data');m=p.metadata
examples,_=load_demonstrations(m['datasets'],json.loads((root/'reports/two-view-20260910-233652/training-configurations.json').read_text()),control_hz=2,goal_supervision=m['goal_supervision'],sample_hz=m['sample_hz'],grasp_goal_height_mm=m.get('grasp_label_height_mm'))
selected=[];episodes=[];counts={}
for e in examples:
    if e['split']!='train':continue
    if e['episode'] not in episodes and len(episodes)<2:episodes.append(e['episode'])
    if e['episode'] in episodes and counts.get(e['episode'],0)<8:
        selected.append(e);counts[e['episode']]=counts.get(e['episode'],0)+1
core=TemporalFeedbackCore(p.circuit)
core.encoder.configure_cameras(p.core.encoder.camera_names,p.core.encoder.camera_rig_revision)
core.encoder.configure(p.core.encoder.profile)
for name in ['retinal_mean','retinal_std','kc_mean','kc_std']:setattr(core,name,getattr(p.core,name).copy())
r=np.asarray([core.encoder.sample(e['observation']['images'],e['directory']) for e in selected]);b=np.asarray([core.body_encoder.sample(e['observation']) for e in selected])
start=time.monotonic();expected=[];last=None
for image,body,e in zip(r,b,selected):
    if e['episode']!=last:core.reset();last=e['episode']
    expected.append(core.response(image,body,simulation_time=e['observation']['simulation_time']))
serial=time.monotonic()-start;expected=np.asarray(expected)
start=time.monotonic();batched=TemporalFeedbackEncoder(core).responses(r,b,selected,batch_size=2);duration=time.monotonic()-start
first=core.encode_responses(expected);second=core.encode_responses(batched)
q1=np.asarray([p.learner.choose(x)[1][:6]*p.joint_gain for x in first]);q2=np.asarray([p.learner.choose(x)[1][:6]*p.joint_gain for x in second])
groups=[[i for i,e in enumerate(selected) if e['episode']==episode] for episode in episodes]
def independent(indices):return TemporalFeedbackEncoder(core).responses(r[indices],b[indices],[selected[i] for i in indices],batch_size=1)
start=time.monotonic()
with ThreadPoolExecutor(max_workers=2) as executor:parts=list(executor.map(independent,groups))
parallel_time=time.monotonic()-start;parallel=np.concatenate(parts)
assert np.array_equal(parallel,expected)
report={'frames':len(selected),'episodes':len(episodes),'serial_seconds':serial,'batch_seconds':duration,'independent_episode_seconds':parallel_time,'parallel_bitwise_match':True,'raw_max_abs_error':float(np.abs(expected-batched).max()),'normalized_max_abs_error':float(np.abs(first-second).max()),'joint_score_max_difference_deg':float(np.abs(q1-q2).max()),'scope':'Same recorded sequence in the live and offline temporal encoders; no robot control'}
assert report['normalized_max_abs_error']<1e-4 and report['joint_score_max_difference_deg']<.05
write_json(run/'temporal-parity.json',report);print(json.dumps(report,indent=2))
