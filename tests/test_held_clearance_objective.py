import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from head_training_runtime import cube_clearance,position_error


def test_clearance_uses_cube_bottom_and_floor():
    position=torch.tensor([[100.,30.,134.]],dtype=torch.float64);rotation=torch.eye(3,dtype=torch.float64)[None]
    value=cube_clearance(position,rotation,torch.zeros((1,3),dtype=torch.float64),rotation,torch.tensor([20.],dtype=torch.float64))
    assert torch.allclose(value,torch.tensor([125.],dtype=torch.float64))


def test_held_clearance_does_not_force_horizontal_position_or_excess_height():
    position=torch.tensor([[400.,30.,150.],[400.,30.,120.]],dtype=torch.float64)
    rotation=torch.eye(3,dtype=torch.float64)[None].repeat(2,1,1)
    data={'goal_position_mm':torch.zeros_like(position),'relative_cube_position_mm':torch.zeros_like(position),
          'relative_cube_rotation':rotation,'cube_size_mm':torch.tensor([20.,20.],dtype=torch.float64)}
    value=position_error(position,rotation,data,torch.arange(2),torch.tensor([True,True]),{'held_position_objective':'clearance','held_clearance_target_mm':125.})
    assert value[0]==0
    assert value[1]==196
    ordinary=position_error(position,rotation,data,torch.arange(2),torch.tensor([False,False]),{'held_position_objective':'clearance','held_clearance_target_mm':125.})
    assert torch.equal(ordinary,position.square().sum(1))
