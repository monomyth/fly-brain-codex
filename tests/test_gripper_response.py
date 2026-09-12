import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('probability_gripper_test',Path(__file__).resolve().parents[1]/'scripts/probability_gripper.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_neural_aperture_is_bounded_continuous_and_monotonic():
    values=[module.neural_aperture(x) for x in [-1,-.001,-1e-6,0,1e-6,.001,1]]
    assert all(0<=x<=90 for x in values)
    assert values==sorted(values) and values[3]==45
    assert values[4]-values[2]<.1
    assert values[0]<.001 and values[-1]>89.999
    with pytest.raises(ValueError):module.neural_aperture(float('nan'))


def test_continuous_grip_preserves_six_neural_joint_targets(monkeypatch):
    joints=[1,2,3,4,5,6]
    monkeypatch.setattr(module.MotorPolicy,'act',lambda self,obs,explore=False:{'joints_deg':joints,'gripper_mm':90})
    policy=object.__new__(module.ProbabilityGripPolicy)
    policy.last_scores=np.array([1,2,3,4,5,6,-.0001]);policy.last_choices=np.zeros(7,dtype=int)
    result=policy.act({'gripper_mm':60,'cube_pose':{'position_mm':[999,999,999]}})
    assert result['joints_deg'] is joints
    assert result['gripper_mm']==module.neural_aperture(-.0001)
    assert policy.last_choices[6]==0
