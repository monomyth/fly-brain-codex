"""Recover rejected neural proposals offline; never connects to a simulator."""
import json
from pathlib import Path
import numpy as np
from fly_brain.adapters import JOINT_LOWER, JOINT_UPPER
from fly_brain.assets import write_json
from fly_brain.visual_dopamine.motor_policy import MotorPolicy

root=Path(__file__).resolve().parents[1];run=root/'reports/policy-recovery-20260911-131608'
policy=MotorPolicy(root/'data/checkpoints/rebot'/run.name,root/'data')
records=[];largest=0.
for episode in [0,1]:
    folder=run/'evaluation'/f'episode-{episode:03d}'
    policy.core.reset();previous=None
    for step in sorted(folder.glob('step-*')):
        path=step/'observation.json'
        if not path.exists():continue
        obs=json.loads(path.read_text());features=policy.core.encode_observation(obs,step)
        scores=policy.learner.choose(features)[1]
        action={'joints_deg':np.clip(scores[:6]*policy.joint_gain,JOINT_LOWER,JOINT_UPPER).tolist(),'gripper_mm':90. if scores[6]>=0 else 0.}
        decision=step/'decision.json'
        if decision.exists():
            expected=json.loads(decision.read_text())['action']
            delta=max(abs(a-b) for a,b in zip(action['joints_deg'],expected['joints_deg']))
            largest=max(largest,delta)
            if delta>1e-4 or action['gripper_mm']!=expected['gripper_mm']:raise ValueError('Offline proposal does not match the recorded actor')
        records.append({'episode':episode,'step':step.name,'accepted':decision.exists(),
                        'from_joints':obs['joints_deg'],'from_grip':obs['gripper_mm'],
                        'to_joints':action['joints_deg'],'to_grip':action['gripper_mm'],
                        'previous_target':previous})
        previous=action
out=run/'floor-timing';out.mkdir(exist_ok=True)
write_json(out/'proposals.json',records)
write_json(out/'replay-check.json',{'max_joint_error_deg':largest,'records':len(records),'rejected_proposals':sum(not r['accepted'] for r in records)})
print({'max_joint_error_deg':largest,'records':len(records)},flush=True)
