"""Compare frozen controllers on identical recorded observations; no live control."""
import json
from pathlib import Path
import numpy as np
import torch
from fly_brain.kinematics import ToolKinematics
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.feedback import ScheduledFeedbackEncoder
from fly_brain.assets import write_json

project=Path(__file__).resolve().parents[1]
run=project/'reports/two-view-20260910-233652'
records=[]
for trial in ['taskspace-nominal-live','supported135-nominal-live']:
    for episode in sorted((run/trial).glob('episode-*')):
        for folder in sorted(episode.glob('step-*'))[:16]:
            obs=json.loads((folder/'observation.json').read_text())
            decision=json.loads((folder/'decision.json').read_text())
            records.append({'source':str(folder.relative_to(run)), 'observation':obs,'directory':folder,'decision':decision})
fk=ToolKinematics()
positions,rotations=fk(torch.tensor([r['observation']['joints_deg'] for r in records],dtype=torch.float64))
results=[]
for i,r in enumerate(records):
    results.append({'source':r['source'],'grasp_position_mm':positions[i].tolist(),'aperture_mm':r['observation']['gripper_mm'],
                    'contacts':r['observation']['finger_contacts'],'models':{}})
for name in ['taskspace','supported135']:
    cp=project/f'data/checkpoints/rebot/two-view-v4-{name}-cpu-seed0-20260911'
    p=MotorPolicy(cp,project/'data')
    retinal=np.asarray([p.core.encoder.sample(r['observation']['images'],r['directory']) for r in records])
    body=np.asarray([p.core.body_encoder.sample(r['observation']) for r in records])
    raw=ScheduledFeedbackEncoder(p.core).responses(retinal,body)
    scores=np.asarray([p.learner.choose(x)[1] for x in p.core.encode_responses(raw)])
    q=scores[:,:6]*p.joint_gain
    goals,rotation=fk(torch.tensor(q,dtype=torch.float64))
    for i,result in enumerate(results):
        result['models'][name]={'goal_joints_deg':q[i].tolist(),'goal_position_mm':goals[i].tolist(),'open':bool(scores[i,6]>=0),'grip_logit':float(scores[i,6]*2000)}
report={'scope':'Same-observation offline comparison. FK used for diagnosis only, never actor inference. The final cube poses in decision files are post-action and are not treated as synchronized input geometry.', 'rows':results}
write_json(run/'same-observation-regression.json',report)
for r in results:
    if 'supported135-nominal-live/episode-000' in r['source']:
        print(r['source'].split('/')[-1], 'xyz',np.round(r['grasp_position_mm'],1),'grip',round(r['aperture_mm'],1),'touch',r['contacts'],flush=True)
        for k,v in r['models'].items():print(k,'goal',np.round(v['goal_position_mm'],1),'open',v['open'],'logit',round(v['grip_logit'],2),flush=True)
