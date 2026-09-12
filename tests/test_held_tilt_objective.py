import sys
from pathlib import Path
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from head_training_runtime import orientation_error,posture_weights


def rotation(axis,angle):return torch.tensor(Rotation.from_euler(axis,angle,degrees=True).as_matrix(),dtype=torch.float64)[None]


def test_held_cube_heading_is_free_but_pregrasp_orientation_is_not():
    predicted=rotation('z',90).repeat(2,1,1);target=torch.eye(3,dtype=torch.float64)[None].repeat(2,1,1)
    error=orientation_error(predicted,target,torch.tensor([True,False]),torch.tensor([[0.,0.,1.]]*2,dtype=torch.float64),'cube_up')
    assert error[0]==0
    assert error[1]>1000


def test_tilted_grip_needs_corrective_wrist_orientation():
    corrected=rotation('x',8);relative=(corrected.transpose(1,2)@torch.tensor([0.,0.,1.],dtype=torch.float64).reshape(1,3,1)).squeeze(-1)
    target=torch.eye(3,dtype=torch.float64)[None];held=torch.tensor([True])
    assert orientation_error(corrected,target,held,relative,'cube_up').item()<1e-20
    assert orientation_error(target,target,held,relative,'cube_up').item()>60


def test_held_posture_weight_is_lower_than_pregrasp():
    held=torch.tensor([True,False])
    assert torch.equal(posture_weights(held,{'held_orientation_objective':'cube_up'}),torch.tensor([.01,.04],dtype=torch.float64))
    assert torch.equal(posture_weights(held,{}),torch.tensor([.04,.04],dtype=torch.float64))


def test_legacy_rotation_objective_is_preserved():
    predicted=rotation('x',8);target=torch.eye(3,dtype=torch.float64)[None]
    expected=(predicted-target).square().sum((1,2))/2*(180/torch.pi)**2
    assert torch.equal(orientation_error(predicted,target,torch.tensor([True]),None),expected)
