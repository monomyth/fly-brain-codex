"""Compare optional Metal inference against the saved sensory feature reference.

Run only after the declared CPU simulator comparison finishes. No robot control.
"""
import argparse,json,time,importlib.metadata
from pathlib import Path
import numpy as np
from fly_brain.assets import write_json,sha256_file
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.feedback import FeedbackCore
from fly_brain.visual_dopamine.temporal import TemporalFeedbackCore
from mlx_sensory_backend import MLXFeedbackCore
from probability_gripper import neural_aperture

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit-episodes',type=int);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
policy=MotorPolicy(a.checkpoint,root/'data');core=policy.core;m=policy.metadata
if m.get('temporal_sensory'):other=TemporalFeedbackCore(policy.circuit,core.updates,m['temporal_sensory'])
else:other=FeedbackCore(policy.circuit,core.updates)
other.encoder.configure_cameras(core.encoder.camera_names,core.encoder.camera_rig_revision);other.encoder.configure(core.encoder.profile)
for name in ['retinal_mean','retinal_std','kc_mean','kc_std']:setattr(other,name,getattr(core,name).copy())
gpu=MLXFeedbackCore(other)
rows=json.loads((a.checkpoint/'feature-rows.json').read_text())
with np.load(a.checkpoint/'training-features.npz',allow_pickle=False) as saved:reference=saved['features'].copy()
assert len(rows)==len(reference)
lookup={}
for dataset in m['datasets']:
    for folder in Path(dataset).glob('episode-*'):
        if folder.name in lookup:raise ValueError('Ambiguous episode directory')
        lookup[folder.name]=folder
selected=[];episodes=[]
for i,row in enumerate(rows):
    episode=row['episode']
    if episode not in episodes:
        if a.limit_episodes is not None and len(episodes)>=a.limit_episodes:continue
        episodes.append(episode)
    selected.append((i,row))
captured=set(m.get('captured_feature_episodes',[]))
cpu_full_episode=next((ep for ep in episodes if ep in captured),None)
cache={};gpu_features=[];gpu_ms=[];cpu_ms=[];feature_error=0.;cpu_error=0.;joint_error=0.;grip_aperture_error=0.;grip_flips=0;last=None;within=0
for index,row in selected:
    episode=row['episode'];folder=lookup[episode]
    if episode not in cache:cache[episode]=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    observation=cache[episode][row['step']]['observation']
    if episode!=last:
        gpu.reset()
        if hasattr(core,'reset'):core.reset()
        else:core.state.fill(0)
        last=episode;within=0
        # Compile/warm the kernel before measuring it, then restore a clean reset.
        if not gpu_ms:gpu.encode_observation(observation,folder);gpu.reset()
    if episode==cpu_full_episode or (episodes.index(episode)<2 and within<6):
        start=time.perf_counter();cpu_value=core.encode_observation(observation,folder);cpu_ms.append((time.perf_counter()-start)*1000)
        cpu_error=max(cpu_error,float(np.abs(cpu_value-reference[index]).max()))
    start=time.perf_counter();value=gpu.encode_observation(observation,folder);gpu_ms.append((time.perf_counter()-start)*1000)
    gpu_features.append(value.copy())
    feature_error=max(feature_error,float(np.abs(value-reference[index]).max()))
    policy.select_motor_bank(observation)
    expected=policy.learner.choose(reference[index])[1];actual=policy.learner.choose(value)[1]
    joint_error=max(joint_error,float(np.abs(expected[:6]-actual[:6]).max()*policy.joint_gain))
    grip_aperture_error=max(grip_aperture_error,abs(neural_aperture(expected[6])-neural_aperture(actual[6])))
    grip_flips+=int((expected[6]>=0)!=(actual[6]>=0));within+=1
    if len(gpu_ms)%100==0:print({'frames':len(gpu_ms),'feature_error':feature_error,'joint_error_deg':joint_error,'gripper_flips':grip_flips},flush=True)
sensory_path=a.output.with_suffix('.sensory.npz')
np.savez(sensory_path,features=np.asarray(gpu_features,dtype=np.float32),row_indices=np.asarray([i for i,_ in selected],dtype=np.int64))
report={'saved_sensory_features':str(sensory_path.resolve()),'saved_sensory_features_sha256':sha256_file(sensory_path),'reference_features_sha256':sha256_file(a.checkpoint/'training-features.npz'),'feature_rows_sha256':sha256_file(a.checkpoint/'feature-rows.json'),'checkpoint':str(a.checkpoint.resolve()),'model_sha256':sha256_file(a.checkpoint/'model.npz'),'manifest_sha256':sha256_file(a.checkpoint/'manifest.json'),'backend_source_sha256':sha256_file(Path(__file__).with_name('mlx_sensory_backend.py')),'gripper_response_source_sha256':sha256_file(Path(__file__).with_name('probability_gripper.py')),'scope':'Offline arithmetic comparison against saved sensory features; no robot control','feature_reference_sources':m.get('training_feature_sources',['CPU encoded features']),'full_cpu_checked_episode':cpu_full_episode,
        'mlx_version':importlib.metadata.version('mlx'),'device':gpu.mx.device_info(),'frames':len(gpu_ms),'episodes':len(episodes),
        'cpu_reference_feature_error':cpu_error,'feature_max_abs_error':feature_error,'joint_target_max_abs_error_deg':joint_error,'gripper_decision_flips':grip_flips,'continuous_gripper_max_abs_error_mm':grip_aperture_error,
        'gpu_encode_median_ms':float(np.median(gpu_ms)),'gpu_encode_p95_ms':float(np.percentile(gpu_ms,95)),
        'cpu_encode_median_ms':float(np.median(cpu_ms)),'passed':cpu_error<1e-5 and feature_error<1e-4 and joint_error<.05 and grip_flips==0 and grip_aperture_error<.05}
write_json(a.output,report);print(json.dumps(report,indent=2),flush=True)
if not report['passed']:raise SystemExit('Metal arithmetic failed the declared equivalence thresholds')
