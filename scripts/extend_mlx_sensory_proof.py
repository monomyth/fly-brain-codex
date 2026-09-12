"""Verify added episodes while reusing an immutable, identical-input replay."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
import mlx.core as mx
from fly_brain.assets import checked_files, sha256_file, write_json
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from mlx_sensory_backend import MLXFeedbackCore
from probability_gripper import neural_aperture

p = argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint', 'source-proof', 'output'):
    p.add_argument('--'+name, type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
source = json.loads(a.source_proof.read_text())
old = Path(source['checkpoint']); new = a.checkpoint.resolve()
old_meta = json.loads((old/'manifest.json').read_text())
new_meta = json.loads((new/'manifest.json').read_text())
checked_files(old, old_meta); checked_files(new, new_meta)
if not source['passed'] or sha256_file(old/'model.npz') != source['model_sha256'] or sha256_file(old/'manifest.json') != source['manifest_sha256']:
    raise ValueError('The original replay and checkpoint must remain unchanged')
for key in ('graph_id', 'circuit_id', 'motor_circuit_id', 'feedback_circuit_id', 'observation_schema', 'temporal_sensory', 'learning_rule', 'package_source_hash', 'architecture'):
    if old_meta.get(key) != new_meta.get(key):
        raise ValueError('Replay extension cannot change the encoder or control contract: '+key)
for filename, key in [('mlx_sensory_backend.py', 'backend_source_sha256'), ('probability_gripper.py', 'gripper_response_source_sha256')]:
    if sha256_file(root/'scripts'/filename) != source[key]:
        raise ValueError('The numerical runtime changed')
if importlib.metadata.version('mlx') != source['mlx_version'] or mx.device_info() != source['device']:
    raise ValueError('The MLX version and device must match the original replay')
with np.load(old/'model.npz') as previous, np.load(new/'model.npz') as current:
    for name in ('retinal_mean', 'retinal_std', 'kc_mean', 'kc_std', 'readout', 'gain'):
        if not np.array_equal(previous[name], current[name]):
            raise ValueError('Calibration changed: '+name)
for filename, key in [('feature-rows.json', 'feature_rows_sha256'), ('training-features.npz', 'reference_features_sha256')]:
    if sha256_file(old/filename) != source[key]:
        raise ValueError('Original replay inputs changed')
cache_path = Path(source['saved_sensory_features'])
if sha256_file(cache_path) != source['saved_sensory_features_sha256']:
    raise ValueError('Original replay cache changed')
old_rows = json.loads((old/'feature-rows.json').read_text())
rows = json.loads((new/'feature-rows.json').read_text())
with np.load(old/'training-features.npz') as z: old_reference = z['features'].copy()
with np.load(new/'training-features.npz') as z: reference = z['features'].copy()
with np.load(cache_path) as z:
    old_replay = z['features'].copy(); indices = z['row_indices'].copy()
if not np.array_equal(indices, np.arange(len(old_rows))) or old_replay.shape != old_reference.shape:
    raise ValueError('An original complete replay is required')
lookup = {(r['episode'], r['step']): i for i, r in enumerate(old_rows)}
new_keys = [(r['episode'], r['step']) for r in rows]
if len(set(new_keys)) != len(rows) or not set(lookup).issubset(new_keys):
    raise ValueError('Rows must be unique and retain all original observations')
old_episodes = {r['episode'] for r in old_rows}
replayed = np.empty_like(reference); added = []
for i, row in enumerate(rows):
    key = new_keys[i]
    if key in lookup:
        j = lookup[key]
        if row != old_rows[j] or not np.array_equal(reference[i], old_reference[j]):
            raise ValueError('An original observation, feature or split changed')
        replayed[i] = old_replay[j]
    else:
        if row['episode'] in old_episodes:
            raise ValueError('New frames in an old episode need its full temporal replay')
        added.append(i)
if not added: raise ValueError('Use the head-only check when there are no added episodes')
policy = MotorPolicy(new, root/'data'); cpu = policy.core
independent = MotorPolicy(new, root/'data'); gpu = MLXFeedbackCore(independent.core)
folders = {f.name:f for d in new_meta['datasets'] for f in Path(d).glob('episode-*')}
records = {}; last = None; seen = set(); gpu_ms = []; cpu_ms = []
cpu_episode = rows[added[0]]['episode']; cpu_error = 0.
for i in added:
    row = rows[i]; ep = row['episode']; folder = folders[ep]
    if ep != last:
        if ep in seen: raise ValueError('Temporal episodes must be contiguous')
        seen.add(ep); last = ep; gpu.reset(); cpu.reset()
        records[ep] = [json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    obs = records[ep][row['step']]['observation']
    if ep == cpu_episode:
        start = time.perf_counter(); value = cpu.encode_observation(obs, folder)
        cpu_ms.append(1000*(time.perf_counter()-start))
        cpu_error = max(cpu_error, float(abs(value-reference[i]).max()))
    start = time.perf_counter(); replayed[i] = gpu.encode_observation(obs, folder)
    gpu_ms.append(1000*(time.perf_counter()-start))
    if len(gpu_ms)%100 == 0: print(json.dumps({'added_frames':len(gpu_ms),'total_added':len(added)}), flush=True)
joint_error = 0.; grip_error = 0.; flips = 0
for recorded, replay in zip(reference, replayed):
    x = policy.learner.choose(recorded)[1]; y = policy.learner.choose(replay)[1]
    joint_error = max(joint_error, float(abs(x[:6]-y[:6]).max()*policy.joint_gain))
    grip_error = max(grip_error, abs(neural_aperture(x[6])-neural_aperture(y[6])))
    flips += int((x[6]>=0)!=(y[6]>=0))
feature_error = float(abs(reference-replayed).max())
path = a.output.with_suffix('.sensory.npz')
np.savez(path, features=replayed, row_indices=np.arange(len(rows)))
report = {**source,
    'checkpoint':str(new),'model_sha256':sha256_file(new/'model.npz'),'manifest_sha256':sha256_file(new/'manifest.json'),
    'reference_features_sha256':sha256_file(new/'training-features.npz'),'feature_rows_sha256':sha256_file(new/'feature-rows.json'),
    'saved_sensory_features':str(path.resolve()),'saved_sensory_features_sha256':sha256_file(path),
    'inherited_replay':str(a.source_proof.resolve()),'inherited_replay_sha256':sha256_file(a.source_proof),
    'verification_source_sha256':sha256_file(__file__),
    'scope':'All original rows reuse an immutable matching replay; every added episode is replayed from images, including one complete independent CPU check. Head outputs are compared on all rows. Timing statistics describe added frames only.',
    'frames':len(rows),'episodes':len({r['episode'] for r in rows}),'reused_frames':len(old_rows),'newly_replayed_frames':len(added),
    'full_cpu_checked_episode':cpu_episode,'cpu_reference_feature_error':max(cpu_error,source['cpu_reference_feature_error']),
    'feature_max_abs_error':feature_error,'joint_target_max_abs_error_deg':joint_error,
    'continuous_gripper_max_abs_error_mm':grip_error,'gripper_decision_flips':flips,
    'gpu_encode_median_ms':float(np.median(gpu_ms)),'gpu_encode_p95_ms':float(np.percentile(gpu_ms,95)),
    'cpu_encode_median_ms':float(np.median(cpu_ms)),
    'passed':cpu_error<1e-5 and feature_error<1e-4 and joint_error<.05 and grip_error<.05 and flips==0}
write_json(a.output, report); print(json.dumps(report, indent=2), flush=True)
if not report['passed']: raise SystemExit('The extended replay failed numerical equivalence')
