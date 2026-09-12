"""Import a verified trained head into a new local checkpoint, retaining sensory calibration."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from fly_brain.assets import checked_files, sha256_file, write_json, artifact_manifest
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.motor_training import save_checkpoint

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--result', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
if a.output.exists() or not a.output.resolve().is_relative_to(root/'data/checkpoints'):
    raise ValueError('Use a new project-local checkpoint directory')
checked_files(a.result, json.loads((a.result/'manifest.json').read_text()))
result = json.loads((a.result/'result.json').read_text())
if sha256_file(a.source/'model.npz') != result['source_model_sha256']:
    raise ValueError('Continuation was fitted from a different source checkpoint')
if not result['fixed_buffers_unchanged'] or not result['parameter_bounds_passed']:
    raise ValueError('Training did not preserve anatomical constraints')
policy = MotorPolicy(a.source, root/'data')
with np.load(a.result/'learned-head.npz', allow_pickle=False) as saved:
    for name in ('weights', 'excitability', 'motor_synapses'):
        value = saved[name]
        if not np.isfinite(value).all():
            raise ValueError('Invalid learned parameters')
        target = policy.circuit.motor_from_dn_pm.data if name == 'motor_synapses' else getattr(policy.learner, name)
        if target.shape != value.shape:
            raise ValueError('Parameter shape mismatch: '+name)
        target[:] = value
metadata = dict(policy.metadata)
# These metrics describe the parent fit and must not be inherited as current results.
for name in ('stage_metrics', 'initial_pose_metrics'):
    metadata.pop(name, None)
metadata.update(training_loss_recipe=result.get('loss_recipe',{}),metrics=result['metrics'], selection_statistics=result['selection_statistics'],
                optimizer_iterations=result['optimizer_iterations'], optimizer_message=result['optimizer_message'],
                training_device=result.get('training_device','cuda'), closed_loop_validated=False, preparation_only=False,
                continuation={'source_checkpoint': str(a.source.resolve()), 'result': str(a.result.resolve()),
                              'source_model_sha256': result['source_model_sha256'],
                              'iterations_requested': result['iterations_requested'],
                              'trainer_sha256': result['trainer_sha256'],
                              'source_bundle_manifest_sha256': result['source_bundle_manifest_sha256'],
                              'elapsed_seconds': result['elapsed_seconds'], 'selection': result['selection']})
save_checkpoint(a.output, policy.core, policy.learner, metadata)
for name in ('training-features.npz','training-targets.npz','parameter-scaling.npy','feature-rows.json'):
    shutil.copy2(a.source/name, a.output/name)
shutil.copy2(a.result/'training-history.json', a.output/'training-history.json')
shutil.copy2(a.result/'result.json', a.output/'continuation-result.json')
for name in ['orientation-supervision.npz','retention-supervision.npz']:
    if (a.result/name).exists():shutil.copy2(a.result/name,a.output/name)
meta = json.loads((a.output/'manifest.json').read_text())
meta.pop('files', None)
artifact_manifest(a.output, meta)
# Loading the artifact checks signs, masks, bounds, readout, and all file hashes.
verified = MotorPolicy(a.output, root/'data')
with np.load(a.output/'model.npz', allow_pickle=False) as actual, np.load(a.source/'model.npz', allow_pickle=False) as before:
    for name in before.files:
        if name not in ('weights','excitability','motor_synapses') and not np.array_equal(actual[name], before[name]):
            raise AssertionError('A fixed checkpoint field changed: '+name)
write_json(a.result/'local-import.json', {'checkpoint': str(a.output.resolve()),
           'model_sha256': sha256_file(a.output/'model.npz'), 'passed': True,
           'fixed_calibration_and_readout_unchanged': True})
print(json.dumps({'checkpoint': str(a.output.resolve()), 'passed': True}), flush=True)
