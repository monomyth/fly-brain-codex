"""Offline diagnostic of information in fixed MaleCNS sensory features.

This unconstrained head is never connected to the simulator and is not a
MaleCNS motor controller. It separates sensory limitations from head limitations.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from fly_brain.kinematics import ToolKinematics,demonstration_target_poses
from fly_brain.visual_dopamine.trajectory_training import load_demonstrations
from fly_brain.assets import write_json

root=Path(__file__).resolve().parents[1];run=root/'reports/two-view-20260910-233652'
checkpoint=root/'data/checkpoints/rebot/two-view-v4-coordinated125-cpu-seed0-20260911'
meta=json.loads((checkpoint/'manifest.json').read_text())
examples,_=load_demonstrations(meta['datasets'],json.loads((run/'training-configurations.json').read_text()),control_hz=2,goal_supervision=meta['goal_supervision'],sample_hz=meta['sample_hz'],grasp_goal_height_mm=meta.get('grasp_label_height_mm'))
with np.load(checkpoint/'training-features.npz',allow_pickle=False) as f:features=f['features'].copy()
expected=[{k:e[k] for k in ('episode','step','split','stage')} for e in examples]
assert expected==json.loads((checkpoint/'feature-rows.json').read_text())
torch.set_num_threads(2);torch.manual_seed(0)
x=torch.tensor(features,dtype=torch.float64)
y=torch.tensor(np.array([e['target'] for e in examples]),dtype=torch.float64)
rows={split:torch.tensor([i for i,e in enumerate(examples) if e['split']==split]) for split in ['train','validation','test']}
fk=ToolKinematics()
with torch.no_grad(): target_position,target_rotation=demonstration_target_poses(fk,y[:,:6],examples,meta['desired_lift_clearance_mm'],0,meta['grasp_goal_height_mm'],meta['hold_center_mm'])
model=nn.Sequential(nn.Linear(x.shape[1],96),nn.Tanh(),nn.Linear(96,64),nn.Tanh(),nn.Linear(64,7)).double()
with torch.no_grad():model[-1].weight.zero_();model[-1].bias.zero_()
center=y[rows['train'],:6].mean(0)
weights=torch.tensor([8. if e['stage'] in ('descend','close') else 1. for e in examples],dtype=torch.float64)
optimizer=torch.optim.Adam(model.parameters(),lr=.001)
best=None;best_value=float('inf');history=[]

def outputs(indices):
    value=model(x[indices]);return center+20*value[:,:6],value[:,6]

def stats(indices):
    with torch.no_grad():
        q,grip=outputs(indices);position,rotation=fk(q)
        angular=(rotation-target_rotation[indices]).square().sum((1,2))/2*(180/torch.pi)**2
        return {'goal_position_rmse_mm':float(torch.sqrt((position-target_position[indices]).square().sum(1).mean())),
                'goal_orientation_rmse_deg':float(torch.sqrt(angular.mean())),
                'gripper_accuracy':float(((grip>=0)==(y[indices,6]>.5)).double().mean())}

for step in range(1601):
    if step%100==0:
        validation=stats(rows['validation']);score=validation['goal_position_rmse_mm']/5+validation['goal_orientation_rmse_deg']/5+3*(1-validation['gripper_accuracy'])
        if score<best_value:best_value=score;best={k:v.detach().clone() for k,v in model.state_dict().items()}
        entry={'step':step,'validation':validation};history.append(entry);print(entry,flush=True)
    if step==1600:break
    selected=rows['train'][torch.randint(len(rows['train']),(128,))]
    q,grip=outputs(selected);position,rotation=fk(q)
    error=(position-target_position[selected]).square().sum(1)
    angular=(rotation-target_rotation[selected]).square().sum((1,2))/2*(180/torch.pi)**2
    posture=(q-y[selected,:6]).square().mean(1)
    classification=nn.functional.binary_cross_entropy_with_logits(grip,y[selected,6],reduction='none')
    loss=(((error+angular)/25+.04*posture+2*classification)*weights[selected]).mean()
    optimizer.zero_grad();loss.backward();optimizer.step()
model.load_state_dict(best)
report={'scope':'Offline information-capacity diagnostic only; unconstrained MLP head, not a MaleCNS motor controller and never deployed to the simulator.',
        'sensory_checkpoint':str(checkpoint),'architecture':'265 -> 96 tanh -> 64 tanh -> 7; train-only output centering',
        'selection':'validation only','metrics':{k:stats(v) for k,v in rows.items()},'history':history}
write_json(run/'sensory-capacity-probe.json',report)
np.savez(run/'sensory-capacity-probe-weights.npz',**{k:v.numpy() for k,v in best.items()})
print(json.dumps(report['metrics'],indent=2),flush=True)
