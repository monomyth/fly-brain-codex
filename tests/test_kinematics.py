import torch
from fly_brain.kinematics import ToolKinematics


def test_folded_and_grasp_positions_match_native_reference():
    fk=ToolKinematics()
    q=torch.tensor([[0.,0,0,0,0,0],[-0.00000253386,-115.2926065101,-74.8726209875,49.5801514942,-0.00021046937,0.00551658437]],dtype=torch.float64)
    position,rotation=fk(q)
    assert torch.allclose(position[0],torch.tensor([240.30584,0,191.70054],dtype=torch.float64),atol=.001)
    assert torch.allclose(position[1],torch.tensor([350.,0,21.],dtype=torch.float64),atol=.025)
    identity=torch.eye(3,dtype=torch.float64)
    assert torch.allclose(rotation@rotation.transpose(1,2),identity,atol=1e-10)


def test_task_space_supervision_has_correct_joint_gradients():
    fk=ToolKinematics()
    q=torch.tensor([[5.,-100,-90,50,10,-2]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(fk,(q,),eps=1e-5,atol=1e-5,rtol=1e-4)


def test_lift_margin_requires_task_space_training_and_stays_bounded(tmp_path):
    import pytest
    from fly_brain.visual_dopamine.feedback_training import train
    for offset in [-1,51,float('nan')]:
        with pytest.raises(ValueError,match='lift target offset'):
            train([],tmp_path/'invalid',{},root=tmp_path,output_mode='joint_targets',task_space_loss=True,lift_goal_offset_mm=offset)
    with pytest.raises(ValueError,match='lift target offset'):
        train([],tmp_path/'invalid',{},root=tmp_path,output_mode='joint_targets',lift_goal_offset_mm=35)


def test_grasp_weighting_preserves_initial_priority_and_ignores_validation_for_normalization():
    import numpy as np
    from fly_brain.visual_dopamine.feedback_training import demonstration_weights
    def row(stage,folded=False):
        return {'stage':stage,'observation':{'joints_deg':[0.]*6 if folded else [0.,-100.,-90.,50.,0.,0.],'gripper_mm':0. if folded else 90.}}
    train=[row('approach',True),row('descend'),row('close'),row('hold')]
    weights=demonstration_weights(train,[0,1,2,3],8,32,8)
    assert np.allclose(weights/weights[-1],[32,8,8,1])
    assert np.isclose(weights.mean(),1.)
    extended=demonstration_weights(train+[row('approach',True)]*20,[0,1,2,3],8,32,8)
    assert np.array_equal(weights,extended[:4])
    baseline=demonstration_weights(train,[0,1,2,3],8,32)
    assert np.allclose(baseline/baseline[-1],[32,1,1,1])


def test_training_grasp_height_follows_goal_role_without_moving_the_approach_goal():
    from fly_brain.kinematics import demonstration_target_poses
    fk=ToolKinematics()
    joints=torch.zeros((3,6),dtype=torch.float64)
    examples=[{'stage':'approach','goal_role':'approach','reference_lift_clearance_mm':115},
              {'stage':'approach','goal_role':'grasp','reference_lift_clearance_mm':115},
              {'stage':'hold','goal_role':'transport','reference_lift_clearance_mm':115}]
    original,rotation=fk(joints)
    actual,actual_rotation=demonstration_target_poses(fk,joints,examples,125,0,27)
    assert torch.equal(actual[0],original[0])
    assert actual[1,2]==27
    assert torch.equal(actual[1,:2],original[1,:2])
    assert actual[2,2]==original[2,2]+10
    assert torch.equal(actual_rotation,rotation)


def test_hold_center_restores_horizontal_position_only_for_transport():
    from fly_brain.kinematics import demonstration_target_poses
    fk=ToolKinematics();q=torch.zeros((2,6),dtype=torch.float64)
    examples=[{'stage':'close','goal_role':'grasp','reference_lift_clearance_mm':115},
              {'stage':'lift','goal_role':'transport','reference_lift_clearance_mm':115}]
    before,rotation=fk(q)
    after,r=demonstration_target_poses(fk,q,examples,125,0,27,[350,0])
    assert torch.equal(after[0,:2],before[0,:2])
    assert torch.equal(after[1,:2],torch.tensor([350.,0.],dtype=torch.float64))
    assert after[0,2]==27
    assert after[1,2]==before[1,2]+10
    assert torch.equal(r,rotation)


def test_common_approach_does_not_replace_cube_specific_grasp_targets():
    from fly_brain.kinematics import demonstration_target_poses
    fk=ToolKinematics();q=torch.zeros((2,6),dtype=torch.float64)
    examples=[{'stage':'approach','goal_role':'approach','reference_lift_clearance_mm':115},
              {'stage':'approach','goal_role':'grasp','reference_lift_clearance_mm':115}]
    before,rotation=fk(q)
    after,r=demonstration_target_poses(fk,q,examples,grasp_goal_height_mm=27,approach_center_mm=[350,0])
    assert torch.equal(after[0,:2],torch.tensor([350.,0.],dtype=torch.float64))
    assert after[0,2]==before[0,2]
    assert torch.equal(after[1,:2],before[1,:2])
    assert after[1,2]==27
    assert torch.equal(r,rotation)
