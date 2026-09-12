"""Summarize recorded commands and native outcomes without changing the actor."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from fly_brain.assets import write_json
from fly_brain.kinematics import ToolKinematics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();data=json.loads(a.evaluation.read_text());fk=ToolKinematics().double();torch.set_num_threads(2)
results=[]
for index,episode in enumerate(data['episodes']):
    steps=episode['steps']
    if not steps:continue
    with torch.no_grad():
        actual,_=fk(torch.tensor([s['actual_joints_deg'] for s in steps],dtype=torch.float64))
        goal,_=fk(torch.tensor([s['action']['joints_deg'] for s in steps],dtype=torch.float64))
    actual,goal=actual.numpy(),goal.numpy()
    cube=np.array([s['evaluation']['cube_pose']['position_mm'] for s in steps])
    xy=np.linalg.norm(actual[:,:2]-cube[:,:2],axis=1)
    contacts=[all('finger_'+side+'_link' in s['evaluation']['contacts'] for side in ['left','right']) for s in steps]
    tail=min(len(steps),40)
    result={'trial':index+1,'success':episode['success'],'steps':len(steps),'error':episode.get('error'),
            'stop_reason':episode.get('stop_reason'),'bilateral_contact_steps':sum(contacts),
            'held_steps':sum(bool(s['evaluation']['held']) for s in steps),
            'max_clearance_mm':max(s['evaluation']['clearance_mm'] for s in steps),
            'max_hold_seconds':max(s['evaluation']['hold_seconds'] for s in steps),
            'floor_limited_steps':sum(s['floor_limited'] for s in steps),
            'tail_frames':tail,'tail_actual_tool_z_mm_median':float(np.median(actual[-tail:,2])),
            'tail_commanded_tool_z_mm_median':float(np.median(goal[-tail:,2])),
            'tail_xy_to_cube_mm_median':float(np.median(xy[-tail:])),
            'tail_actual_aperture_mm_median':float(np.median([s['actual_gripper_mm'] for s in steps[-tail:]])),
            'tail_commanded_aperture_mm_median':float(np.median([s['action']['gripper_mm'] for s in steps[-tail:]])),
            'inference_ms_median':float(np.median([s['timings_ms']['inference'] for s in steps]))}
    results.append(result)
write_json(a.output,{'source':str(a.evaluation.resolve()),'note':'FK of the recorded native joint state and commanded goal; goal is not the next physical pose. Cube geometry is evaluator-only. Tail medians summarize the last 40 decisions, not a fixed time interval.','episodes':results})
print(json.dumps(results,indent=2))
