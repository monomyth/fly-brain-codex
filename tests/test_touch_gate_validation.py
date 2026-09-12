import sys,json
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from touch_gate_validation import validate_touch_gate,ROOT
from fly_brain.assets import sha256_file


def test_legacy_deployment_does_not_require_a_second_response_proof(tmp_path):
    validate_touch_gate(tmp_path,{}, {'name':'single','touch_gated_hold':None})


def test_new_response_proof_is_required_and_bound_to_both_weight_files(tmp_path):
    for name,value in [('model.npz',b'pickup'),('holding-model.npz',b'holding'),('manifest.json',b'{}')]:
        (tmp_path/name).write_bytes(value)
    metadata={'touch_gated_hold':{'version':1}}
    with pytest.raises(ValueError,match='verification'):
        validate_touch_gate(tmp_path,{'frames':2},metadata)
    result={'passed':True,'action_max_abs_error':0.,'frames':2,'bank_frames':{'pickup':1,'holding':1},
            'model_sha256':sha256_file(tmp_path/'model.npz'),'manifest_sha256':sha256_file(tmp_path/'manifest.json'),
            'holding_model_sha256':sha256_file(tmp_path/'holding-model.npz'),
            'motor_policy_source_sha256':sha256_file(ROOT/'src/fly_brain/visual_dopamine/motor_policy.py')}
    proof={'frames':2,'touch_gate_verification':result}
    validate_touch_gate(tmp_path,proof,metadata)
    (tmp_path/'holding-model.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='verification'):validate_touch_gate(tmp_path,proof,metadata)
