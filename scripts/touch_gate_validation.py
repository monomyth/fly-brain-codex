"""Runtime evidence checks for tactile-selected learned response banks."""
import json
from pathlib import Path
from fly_brain.assets import sha256_file

ROOT=Path(__file__).resolve().parents[1]


def validate_touch_gate(checkpoint,proof,manifest=None):
    checkpoint=Path(checkpoint)
    manifest=manifest or json.loads((checkpoint/'manifest.json').read_text())
    if not manifest.get('touch_gated_hold'):
        return
    result=proof.get('touch_gate_verification',{})
    expected={'model_sha256':sha256_file(checkpoint/'model.npz'),
              'manifest_sha256':sha256_file(checkpoint/'manifest.json'),
              'holding_model_sha256':sha256_file(checkpoint/'holding-model.npz'),
              'motor_policy_source_sha256':sha256_file(ROOT/'src/fly_brain/visual_dopamine/motor_policy.py')}
    counts=result.get('bank_frames',{})
    if (not result.get('passed') or result.get('action_max_abs_error')!=0
            or result.get('frames')!=proof['frames']
            or any(result.get(key)!=value for key,value in expected.items())
            or counts.get('pickup',0)<=0 or counts.get('holding',0)<=0
            or sum(counts.values())!=proof['frames']):
        raise ValueError('The touch-selected controller needs matching complete response-bank verification')
