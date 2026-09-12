import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_mlx_ui import validate_deployment,ROOT
from fly_brain.assets import sha256_file,write_json,artifact_manifest


def deployment(tmp_path,successes=18):
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    (checkpoint/'model.npz').write_bytes(b'fixture')
    write_json(checkpoint/'feature-rows.json',[{'episode':'recording','step':0}])
    artifact_manifest(checkpoint,{'schema':'test','architecture':{'control_hz':2.}})
    hashes={'model_sha256':sha256_file(checkpoint/'model.npz'),'manifest_sha256':sha256_file(checkpoint/'manifest.json'),
            'backend_source_sha256':sha256_file(ROOT/'scripts/mlx_sensory_backend.py'),
            'gripper_response_source_sha256':sha256_file(ROOT/'scripts/probability_gripper.py')}
    proof=tmp_path/'parity.json'
    write_json(proof,{'passed':True,'checkpoint':str(checkpoint),'frames':1,'episodes':1,**hashes})
    evaluation=tmp_path/'evaluation';evaluation.mkdir()
    write_json(evaluation/'runtime.json',{'gripper_response':'probability','learn':False,'max_steps':160,'settle_seconds':.1,'control_hz_override':None,**hashes})
    task={'cube_size_mm':20,'initial_joints_deg':[0]*6,'initial_gripper_mm':0}
    episodes=[{'success':i<successes,'trial_started':True,'task_for_evaluator_only':task,
               'final_evaluation':{'success':i<successes,'held':True,'hold_seconds':5.01,'clearance_mm':110,'tilt_deg':2}} for i in range(20)]
    write_json(evaluation/'evaluation.json',{'checkpoint':str(checkpoint),'learning_enabled':False,'attempts':20,'episodes':episodes})
    return {'checkpoint':str(checkpoint),'parity':str(proof),'qualification':str(evaluation/'evaluation.json')}


def test_requires_full_qualified_native_results(tmp_path):
    settings=deployment(tmp_path)
    assert validate_deployment(settings)['qualification_successes']==18
    path=Path(settings['qualification']);value=json.loads(path.read_text());value['episodes']=value['episodes'][:19];value['attempts']=19
    write_json(path,value)
    with pytest.raises(ValueError,match='completed 20-trial'):validate_deployment(settings)


def test_rejects_insufficient_native_success(tmp_path):
    with pytest.raises(ValueError,match='18/20'):validate_deployment(deployment(tmp_path,17))


def test_rejects_wrong_runtime_even_with_good_score(tmp_path):
    settings=deployment(tmp_path);path=Path(settings['qualification']).with_name('runtime.json')
    value=json.loads(path.read_text());value['gripper_response']='binary';write_json(path,value)
    with pytest.raises(ValueError,match='different controller runtime'):validate_deployment(settings)


def test_uses_native_pose_metrics_not_only_success_flags(tmp_path):
    settings=deployment(tmp_path,20);path=Path(settings['qualification']);value=json.loads(path.read_text())
    for row in value['episodes'][:3]:row['final_evaluation']['tilt_deg']=8
    write_json(path,value)
    with pytest.raises(ValueError,match='18/20'):validate_deployment(settings)


def test_rejects_unmatched_timing(tmp_path):
    settings=deployment(tmp_path);path=Path(settings['qualification']).with_name('runtime.json')
    value=json.loads(path.read_text());value['control_hz_override']=4.;write_json(path,value)
    with pytest.raises(ValueError,match='timing profile'):validate_deployment(settings)
