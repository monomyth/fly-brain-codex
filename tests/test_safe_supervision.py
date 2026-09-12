import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from safe_supervision import BatchKinematics, safe_goal
from fly_brain.kinematics import ToolKinematics


def observation(position, cube=(350., 50., 9.), contacts=(False, False)):
    return {
        "grasp_pose": {"position_mm": list(position), "quaternion_xyzw": Rotation.from_euler("y", 90, degrees=True).as_quat().tolist()},
        "cube_pose": {"position_mm": list(cube), "quaternion_xyzw": [0., 0., 0., 1.]},
        "finger_contacts": dict(zip(("left", "right"), contacts)),
    }


def test_misaligned_grasp_raises_before_sideways_travel():
    obs = observation((350., 0., 27.))
    position, _, aperture, role = safe_goal(obs, {"cube_pose": obs["cube_pose"]})
    assert role == "raise" and aperture == 90.
    np.testing.assert_allclose(position, [350., 0., 65.])
    obs["grasp_pose"]["position_mm"] = [350., 0., 65.]
    position, _, aperture, role = safe_goal(obs, {"cube_pose": obs["cube_pose"]})
    assert role == "align" and aperture == 90.
    np.testing.assert_allclose(position, [350., 50., 65.])


def test_close_requires_lateral_and_vertical_alignment():
    for position, expected in [((350, 50, 65), "descend"), ((350, 50, 27), "close"), ((350, 55, 27), "raise")]:
        obs = observation(position)
        _, _, aperture, role = safe_goal(obs, {"cube_pose": obs["cube_pose"]})
        assert role == expected
        assert (aperture == 0.) == (expected == "close")


def test_stable_grasp_holds_current_pose():
    obs = observation((360., 50., 151.), cube=(360., 50., 133.), contacts=(True, True))
    position, rotation, aperture, role = safe_goal(obs, {"cube_pose": obs["cube_pose"], "clearance_mm": 124., "tilt_deg": 2.})
    assert role == "hold" and aperture == 0.
    np.testing.assert_allclose(position, obs["grasp_pose"]["position_mm"])


def test_batch_kinematics_matches_training_fk_and_finite_difference():
    q = np.array([[0., 0., 0., 0., 0., 0.], [10., -90., -80., 20., 5., 0.]])
    kin = BatchKinematics()
    position, rotation, jacobian = kin.forward(q)
    expected_position, expected_rotation = ToolKinematics().double()(torch.tensor(q))
    np.testing.assert_allclose(position, expected_position.detach().numpy(), atol=1e-8)
    np.testing.assert_allclose(rotation, expected_rotation.detach().numpy(), atol=1e-10)
    for joint in range(6):
        shifted = q.copy(); shifted[:, joint] += 1e-4
        next_position, next_rotation, _ = kin.forward(shifted)
        np.testing.assert_allclose((next_position-position)/1e-4, jacobian[:, :3, joint], atol=1e-5)
        angular = Rotation.from_matrix(next_rotation@rotation.transpose(0, 2, 1)).as_rotvec()/1e-4
        np.testing.assert_allclose(angular, jacobian[:, 3:, joint], atol=1e-8)


def test_smooth_descent_has_no_height_jump_at_alignment_boundary():
    targets = []
    for offset in (2.9999, 3.0001, 5., 10.):
        obs = observation((350., 50.+offset, 60.))
        position, _, aperture, _ = safe_goal(obs, {"cube_pose": obs["cube_pose"]}, "smooth")
        targets.append(position[2])
        assert aperture == 90.
    assert abs(targets[1]-targets[0]) < 1e-6
    assert 27. < targets[2] < 65.
    assert targets[3] == 65.


def test_fixed_hold_target_restores_height_and_position_after_drift():
    obs = observation((365., 55., 170.), cube=(365., 55., 152.), contacts=(True, True))
    position, _, aperture, role = safe_goal(obs, {"cube_pose": obs["cube_pose"], "clearance_mm": 143., "tilt_deg": 0.}, "smooth", [350., 0.])
    np.testing.assert_allclose(position, [350., 0., 152.])
    assert aperture == 0. and role == 'lift'


def test_direct_descent_does_not_create_a_near_cube_hover_target():
    obs=observation((350.,55.,42.));obs['gripper_mm']=90.
    position,_,aperture,role=safe_goal(obs,{'cube_pose':obs['cube_pose']},'direct')
    np.testing.assert_allclose(position,[350.,50.,27.])
    assert aperture==90. and role=='descend'


def test_direct_closure_survives_small_pose_fluctuations_but_reopens_an_empty_jaw():
    obs=observation((350.,54.,33.));obs['gripper_mm']=40.
    _,_,aperture,role=safe_goal(obs,{'cube_pose':obs['cube_pose']},'direct')
    assert aperture==0. and role=='close'
    obs['gripper_mm']=0.
    position,_,aperture,role=safe_goal(obs,{'cube_pose':obs['cube_pose']},'direct')
    assert aperture==90. and role=='raise' and position[2]>=65.
