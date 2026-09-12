"""Verify a new neural head against a completed, identical-input sensory replay."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import mlx.core as mx
from fly_brain.assets import sha256_file,checked_files,write_json
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from probability_gripper import neural_aperture

p=argparse.ArgumentParser(description=__doc__)
for key in ['checkpoint','source-proof','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[1]
source=json.loads(a.source_proof.read_text());old=Path(source['checkpoint']);new=a.checkpoint.resolve()
if not source['passed']:raise ValueError('The source replay did not pass')
old_meta=json.loads((old/'manifest.json').read_text());new_meta=json.loads((new/'manifest.json').read_text())
checked_files(old,old_meta);checked_files(new,new_meta)
if sha256_file(old/'model.npz')!=source['model_sha256'] or sha256_file(old/'manifest.json')!=source['manifest_sha256']:
    raise ValueError('Source checkpoint changed after its sensory replay')
for key in ['graph_id','circuit_id','motor_circuit_id','feedback_circuit_id','observation_schema','temporal_sensory','learning_rule','package_source_hash']:
    if old_meta.get(key)!=new_meta.get(key):raise ValueError('Sensory/runtime contract changed: '+key)
decoder_keys={'output_mode','joint_gain','command_deadband_deg','position_servo','command_scale','ramp_gripper','active_joints'}
old_sensory={k:v for k,v in old_meta['architecture'].items() if k not in decoder_keys}
new_sensory={k:v for k,v in new_meta['architecture'].items() if k not in decoder_keys}
if old_sensory!=new_sensory:raise ValueError('Sensory architecture changed')
for file,key in [('training-features.npz','reference_features_sha256'),('feature-rows.json','feature_rows_sha256')]:
    if sha256_file(new/file)!=source.get(key):raise ValueError('Input sequences or reference features changed')
for file,key in [('mlx_sensory_backend.py','backend_source_sha256'),('probability_gripper.py','gripper_response_source_sha256')]:
    if sha256_file(root/'scripts'/file)!=source[key]:raise ValueError('Numerical runtime changed')
if importlib.metadata.version('mlx')!=source['mlx_version'] or mx.device_info()!=source['device']:
    raise ValueError('Replay reuse requires the same MLX version and device')
with np.load(old/'model.npz',allow_pickle=False) as previous,np.load(new/'model.npz',allow_pickle=False) as current:
    for key in ['retinal_mean','retinal_std','kc_mean','kc_std','readout','gain']:
        if not np.array_equal(previous[key],current[key]):raise ValueError('Calibration or readout changed: '+key)
cache=Path(source['saved_sensory_features'])
if sha256_file(cache)!=source['saved_sensory_features_sha256']:raise ValueError('Saved sensory replay checksum mismatch')
with np.load(cache,allow_pickle=False) as saved:replayed=saved['features'].copy();indices=saved['row_indices'].copy()
with np.load(new/'training-features.npz',allow_pickle=False) as saved:reference=saved['features'].copy()
rows=json.loads((new/'feature-rows.json').read_text())
if not np.array_equal(indices,np.arange(len(rows))) or replayed.shape!=reference.shape or source['frames']!=len(rows) or source['episodes']!=len({r['episode'] for r in rows}):
    raise ValueError('A complete, identically ordered replay is required')
policy=MotorPolicy(new,root/'data');joint_error=0.;grip_error=0.;flips=0;binary_grip_error=0.
folders={p.name:p for d in new_meta['datasets'] for p in Path(d).glob('episode-*')};observations={}
class ReplayCore:
    def __init__(self,state):self.state=state;self.features=None
    def encode_observation(self,*args,**kwargs):return self.features
core=ReplayCore(policy.core.state);policy.core=core
for row,cpu,gpu in zip(rows,reference,replayed):
    ep=row['episode']
    if ep not in observations:observations[ep]=[json.loads(line) for line in (folders[ep]/'steps.jsonl').read_text().splitlines()]
    obs=observations[ep][row['step']]['observation'];actions=[];scores=[]
    for features in [cpu,gpu]:
        core.features=features;policy.reset(obs);actions.append(policy.act(obs));scores.append(policy.last_scores.copy())
    joint_error=max(joint_error,float(np.max(np.abs(np.asarray(actions[0]['joints_deg'])-actions[1]['joints_deg']))))
    binary_grip_error=max(binary_grip_error,abs(actions[0]['gripper_mm']-actions[1]['gripper_mm']))
    grip_error=max(grip_error,abs(neural_aperture(scores[0][6])-neural_aperture(scores[1][6])))
    flips+=int((scores[0][6]>=0)!=(scores[1][6]>=0))
report={**source,'checkpoint':str(new),'model_sha256':sha256_file(new/'model.npz'),'manifest_sha256':sha256_file(new/'manifest.json'),
        'reused_sensory_proof':str(a.source_proof.resolve()),'reused_sensory_proof_sha256':sha256_file(a.source_proof),
        'verification_source_sha256':sha256_file(__file__),'scope':'Identical-input sensory replay reused after graph, calibration, code, device and sequence checks; the new neural head was compared on every frame. Encoder timing is inherited from the source replay.',
        'decoder_architecture':new_meta['architecture'],'binary_gripper_max_abs_error_mm':binary_grip_error,'joint_target_max_abs_error_deg':joint_error,'continuous_gripper_max_abs_error_mm':grip_error,'gripper_decision_flips':flips,
        'passed':joint_error<.05 and grip_error<.05 and binary_grip_error<.05 and flips==0}
write_json(a.output,report);print(json.dumps(report,indent=2),flush=True)
if not report['passed']:raise SystemExit('New head failed numerical replay equivalence')
