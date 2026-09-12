"""Arithmetic check at a new observation cadence; no physical robot execution."""
import argparse
import json
from pathlib import Path
import numpy as np
from fly_brain.assets import sha256_file,write_json
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.temporal import TemporalFeedbackCore
from mlx_sensory_backend import MLXFeedbackCore
from probability_gripper import neural_aperture

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--interval',type=float,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[1]
policy=MotorPolicy(a.checkpoint,root/'data');cpu=policy.core
base=TemporalFeedbackCore(policy.circuit,cpu.updates,cpu.temporal_config)
base.encoder.configure_cameras(cpu.encoder.camera_names,cpu.encoder.camera_rig_revision);base.encoder.configure(cpu.encoder.profile)
for name in ['retinal_mean','retinal_std','kc_mean','kc_std']:setattr(base,name,getattr(cpu,name).copy())
gpu=MLXFeedbackCore(base)
rows=json.loads((a.checkpoint/'feature-rows.json').read_text());ids=list(dict.fromkeys(r['episode'] for r in rows))[:2]
folders={f.name:f for dataset in policy.metadata['datasets'] for f in Path(dataset).glob('episode-*')}
feature_error=joint_error=grip_error=0.;count=0
for episode in ids:
    cpu.reset();gpu.reset();folder=folders[episode]
    original=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    for number,item in enumerate(r for r in rows if r['episode']==episode):
        observation={**original[item['step']]['observation'],'simulation_time':number*a.interval}
        reference=cpu.encode_observation(observation,folder);actual=gpu.encode_observation(observation,folder)
        feature_error=max(feature_error,float(np.abs(reference-actual).max()))
        expected=policy.learner.choose(reference)[1];result=policy.learner.choose(actual)[1]
        joint_error=max(joint_error,float(np.abs(expected[:6]-result[:6]).max()*policy.joint_gain))
        grip_error=max(grip_error,abs(neural_aperture(expected[6])-neural_aperture(result[6])))
        count+=1
report={'scope':'Recorded inputs with synthetic uniform timestamps; numerical check only, not a physical trial',
        'interval_seconds':a.interval,'frames':count,'episodes':len(ids),'feature_max_abs_error':feature_error,
        'joint_target_max_abs_error_deg':joint_error,'continuous_gripper_max_abs_error_mm':grip_error,
        'model_sha256':sha256_file(a.checkpoint/'model.npz'),'manifest_sha256':sha256_file(a.checkpoint/'manifest.json'),
        'backend_source_sha256':sha256_file(Path(__file__).with_name('mlx_sensory_backend.py')),
        'gripper_response_source_sha256':sha256_file(Path(__file__).with_name('probability_gripper.py')),
        'passed':feature_error<1e-4 and joint_error<.05 and grip_error<.05}
write_json(a.output,report);print(json.dumps(report,indent=2),flush=True)
if not report['passed']:raise SystemExit('Cadence arithmetic check failed')
