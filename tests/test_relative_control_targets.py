from types import SimpleNamespace
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from head_training_runtime import kinematic_joints


def test_relative_signals_are_bounded_increments_on_encoders():
    scores=torch.tensor([[2.,-1.,0.,0.,0.,0.,0.]],dtype=torch.float64)
    data={'current_joints_deg':torch.tensor([[0.,-90.,-70.,40.,0.,0.]],dtype=torch.float64)}
    actual=kinematic_joints(scores,SimpleNamespace(joint_gain=15),data,torch.tensor([0]),{'output_mode':'joint_deltas','control_hz':2})
    assert torch.equal(actual,torch.tensor([[15.,-97.5,-70.,40.,0.,0.]],dtype=torch.float64))


def test_neutral_relative_joint_signal_holds_current_pose():
    current=torch.tensor([[20.,-95.,-80.,55.,1.,2.]],dtype=torch.float64)
    actual=kinematic_joints(torch.zeros((1,7),dtype=torch.float64),SimpleNamespace(joint_gain=60),{'current_joints_deg':current},torch.tensor([0]),{'output_mode':'joint_deltas','control_hz':2})
    assert torch.equal(actual,current)
