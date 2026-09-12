import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from motor_output_filter import MotorOutputFilter

def action(grip=90,q=0):return {'joints_deg':[q]*6,'gripper_mm':grip}

def test_full_stroke_can_finish_before_reopening():
    f=MotorOutputFilter();f.apply(action(),0)
    assert f.apply(action(0),.5)['gripper_mm']==0
    for t in np.arange(1,4.5,.5):assert f.apply(action(90),t)['gripper_mm']==0
    assert f.apply(action(90),4.5)['gripper_mm']==0
    assert f.apply(action(90),5)['gripper_mm']==0
    assert f.apply(action(90),5.5)['gripper_mm']==90

def test_opening_request_must_persist_after_commitment():
    f=MotorOutputFilter();f.apply(action(0),0)
    f.apply(action(90),4);f.apply(action(40),4.5)
    assert f.apply(action(90),5)['gripper_mm']==0
    assert f.apply(action(90),5.5)['gripper_mm']==0
    assert f.apply(action(90),6)['gripper_mm']==90

def test_episode_reset_does_not_carry_a_close_command():
    f=MotorOutputFilter();f.apply(action(0),1);f.reset()
    assert f.apply(action(90),0)['gripper_mm']==90

def test_joint_integration_preserves_initial_command_and_bounds():
    f=MotorOutputFilter();assert f.apply(action(q=10),0)['joints_deg']==[10]*6
    result=f.apply(action(q=20),.5)
    assert np.allclose(result['joints_deg'],10+(1-np.exp(-.5/.6))*10)
    assert all(10<q<20 for q in result['joints_deg'])

def test_time_reversal_and_nonfinite_commands_are_rejected():
    f=MotorOutputFilter();f.apply(action(),1)
    with pytest.raises(ValueError):f.apply(action(),.5)
    with pytest.raises(ValueError):f.apply(action(q=float('nan')),2)
