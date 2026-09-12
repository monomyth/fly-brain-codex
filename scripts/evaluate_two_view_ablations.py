"""Measure each camera's contribution with frozen model weights on validation frames."""
import json
from pathlib import Path
import numpy as np
from fly_brain.assets import write_json
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.feedback import ScheduledFeedbackEncoder
from fly_brain.visual_dopamine.trajectory_training import load_demonstrations

project=Path(__file__).resolve().parents[1];settings=json.loads((project/'reports/two-view-current.json').read_text());run=Path(settings['run']);plan=json.loads((run/'plan.json').read_text())
policy=MotorPolicy(settings['checkpoint'],project/'data');core=policy.core
examples,_=load_demonstrations(plan['dataset'],plan['configurations'],control_hz=2,goal_supervision='observable',sample_hz=2)
examples=[e for e in examples if e['split']=='validation']
r=np.asarray([core.encoder.sample(e['observation']['images'],e['directory']) for e in examples]);body=np.asarray([core.body_encoder.sample(e['observation']) for e in examples])
targets=np.asarray([e['target'] for e in examples]);initial=np.array([max(abs(np.asarray(e['observation']['joints_deg'])))<1e-6 and e['observation']['gripper_mm']<1e-6 for e in examples])
encoder=ScheduledFeedbackEncoder(core);report={'checkpoint':settings['checkpoint'],'split':'validation','frames':len(examples),'method':'Freeze learned weights and replace one camera retinal drive with its mean (zero contrast); body inputs are preserved','conditions':{}}
for name,side in [('both',None),('without_front',0),('without_gripper',1),('without_images','all')]:
    stimulus=r.copy()
    if side is not None:
        mask=np.ones(len(core.circuit.retina),dtype=bool) if side=='all' else core.circuit.retina_side==side
        stimulus[:,mask]=core.retinal_mean[mask]
    features=core.encode_responses(encoder.responses(stimulus,body))
    scores=np.asarray([policy.learner.choose(x)[1] for x in features]);joints=scores[:,:6]*policy.metadata['architecture']['joint_gain']
    error=joints-targets[:,:6]
    result={'joint_rmse_deg':float(np.sqrt(np.mean(error**2))),'joint_mae_deg':np.mean(abs(error),axis=0).tolist(),'gripper_accuracy':float(np.mean((scores[:,6]>=0)==(targets[:,6]>.5))),
            'initial_joint_mae_deg':np.mean(abs(error[initial]),axis=0).tolist(),'initial_predicted_joints_deg':joints[initial].tolist(),'initial_target_joints_deg':targets[initial,:6].tolist()}
    report['conditions'][name]=result;write_json(run/'camera-ablations.json',report);print({name:result},flush=True)
