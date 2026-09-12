"""Report physical prediction errors with the checkpoint's actual training targets.

Uses recorded teacher geometry for offline diagnosis only. No robot control.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from fly_brain.kinematics import ToolKinematics, demonstration_target_poses
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.trajectory_training import load_demonstrations, MotorTargetNetwork
from fly_brain.adapters import JOINT_LOWER, JOINT_UPPER
from fly_brain.assets import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    project=Path(__file__).resolve().parents[1]
    policy=MotorPolicy(args.checkpoint,project/'data');meta=policy.metadata
    if policy.output_mode!='joint_targets':raise ValueError('This diagnostic requires an absolute joint-target checkpoint')
    run=Path(json.loads((project/'reports/two-view-current.json').read_text())['run'])
    config=json.loads((run/'training-configurations.json').read_text())
    examples,_=load_demonstrations(meta['datasets'],config,control_hz=2,goal_supervision=meta['goal_supervision'],sample_hz=meta.get('sample_hz'),grasp_goal_height_mm=meta.get('grasp_label_height_mm'))
    with np.load(args.checkpoint/'training-features.npz',allow_pickle=False) as saved:features=saved['features'].copy()
    recorded=json.loads((args.checkpoint/'feature-rows.json').read_text())
    if recorded!=[{k:e[k] for k in ('episode','step','split','stage')} for e in examples]:
        raise ValueError('Recorded feature order differs from current demonstration loading')
    model=MotorTargetNetwork(policy.circuit).double()
    with torch.no_grad():
        model.weights.copy_(torch.tensor(policy.learner.weights));model.excitability.copy_(torch.tensor(policy.learner.excitability))
        scores=model(torch.tensor(features,dtype=torch.float64)).numpy()
        predicted=np.clip(scores[:,:6]*policy.joint_gain,JOINT_LOWER,JOINT_UPPER)
        fk=ToolKinematics();actual,actual_rotation=fk(torch.tensor(predicted,dtype=torch.float64))
        expected,expected_rotation=demonstration_target_poses(fk,torch.tensor(np.array([e['target'][:6] for e in examples]),dtype=torch.float64),examples,meta.get('desired_lift_clearance_mm'),meta.get('lift_goal_offset_mm',0.),meta.get('grasp_goal_height_mm'),meta.get('hold_center_mm'),meta.get('approach_center_mm'))
    delta=(actual-expected).numpy()
    angle=Rotation.from_matrix((actual_rotation@expected_rotation.transpose(1,2)).numpy()).magnitude()*180/np.pi
    def statistics(indices):
        distance=np.linalg.norm(delta[indices],axis=1)
        return {'frames':len(indices),'goal_position_rmse_mm':float(np.sqrt(np.mean(distance**2))),
                'goal_lateral_error_p95_mm':float(np.percentile(np.linalg.norm(delta[indices,:2],axis=1),95)),
                'goal_orientation_rmse_deg':float(np.sqrt(np.mean(angle[indices]**2))),
                'gripper_accuracy':float(np.mean((scores[indices,6]>=0)==[examples[i]['target'][6]>.5 for i in indices]))}
    splits={}
    for name in ('train','validation','test'):
        indices=[i for i,e in enumerate(examples) if e['split']==name]
        if indices:
            splits[name]={**statistics(indices),'stages':{stage:statistics([i for i in indices if examples[i]['stage']==stage]) for stage in sorted({examples[i]['stage'] for i in indices})}}
    report={'checkpoint':str(args.checkpoint.resolve()),'splits':splits,'desired_lift_clearance_mm':meta.get('desired_lift_clearance_mm'),
            'approach_center_mm':meta.get('approach_center_mm'),'hold_center_mm':meta.get('hold_center_mm'),'grasp_goal_height_mm':meta.get('grasp_goal_height_mm'),'lift_goal_offset_mm':meta.get('lift_goal_offset_mm',0.),'scope':'Offline FK of clamped neural goals against the declared training targets; not measured execution accuracy.'}
    write_json(args.output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
