from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from fly_brain.dataset import EpisodeWriter, split_dataset
from fly_brain.simulator import configure_episode
from fly_brain.teacher import Teacher


class FakeSimulator:
    def __init__(self):
        self.calls = []

    def release(self):
        pass

    def wait_ready(self):
        pass

    def call(self, tool, **arguments):
        self.calls.append((tool, arguments))
        if tool == "rebot_get_task":
            return {"task": {"cube_xy_mm": [350, 0]}}
        if tool == "rebot_solve_pose":
            return {"joints_deg": [0, -100, -90, 20, 0, 0]}
        return {}


def test_placement_changes_reuse_physics_and_restore_start():
    client = FakeSimulator()
    task = {"cube_xy_mm": [350, 0], "cube_size_mm": 50, "input_mode": "state"}
    configure_episode(client, task)
    configure_episode(client, {**task, "cube_xy_mm": [340, -20]})
    tools = [name for name, _ in client.calls]
    assert tools.count("rebot_configure_task") == 1
    assert tools.index("rebot_reset_episode") < tools.index("rebot_place_cube")
    configure_episode(client, {**task, "friction": .5})
    assert [name for name, _ in client.calls].count("rebot_configure_task") == 2


def test_teacher_does_not_release_before_verified_hold(observation):
    client = FakeSimulator()
    task = {"cube_xy_mm": [350, 0], "cube_size_mm": 50, "cube_yaw_deg": 0,
            "lift_clearance_mm": 100, "hold_seconds": 5}
    teacher = Teacher(client, task)
    teacher.stage = "hold"
    action, label = teacher.act(observation, {"success": False})
    assert action["gripper_mm"] == 0 and label[-1] == 0
    action, label = teacher.act(observation, {"success": True})
    assert action["gripper_mm"] == 90 and label[-1] == 1


def test_new_dagger_episodes_preserve_held_out_groups(tmp_path, observation):
    def append(x, provenance="conventional", success=True):
        writer = EpisodeWriter(tmp_path, {"cube_xy_mm": [x, 0]}, provenance, 5, {"teacher_version": "cube_feedback_v2"})
        writer.append(observation, {"joints_deg": observation["joints_deg"], "gripper_mm": 0}, [0] * 7, "hold")
        writer.finish({"success": success})
    for x in (330, 340, 350, 360):
        append(x)
    before = split_dataset(tmp_path)
    append(370, "teacher_assisted", False)
    append(350, "teacher_assisted", False)
    after = split_dataset(tmp_path, include_dagger_failures=True)
    for key, assignment in before["configuration_split"].items():
        assert after["configuration_split"][key] == assignment
    assert sum(map(len, after["splits"].values())) == 6


def test_floor_projection_preserves_proposed_direction_without_task_state():
    from fly_brain.simulator import floor_limited_action, SimulatorError
    class Client:
        def __init__(self): self.calls = 0
        def action(self, observation, action):
            self.calls += 1
            if action['joints_deg'][0] > 1:
                raise SimulatorError('Requested endpoint intersects the floor.')
            return {'accepted': True}
    client = Client()
    observation = {'joints_deg': [0] * 6, 'gripper_mm': 50}
    proposal = {'joints_deg': [4, 2, 0, 0, 0, 0], 'gripper_mm': 0}
    _, applied, report = floor_limited_action(client, observation, proposal)
    assert applied['joints_deg'][:2] == [1, .5]
    assert applied['gripper_mm'] == 37.5
    assert report['fraction'] == .25 and client.calls == 3


def test_floor_projection_never_retries_uncertain_motion():
    import pytest
    from fly_brain.simulator import floor_limited_action
    class Client:
        def __init__(self): self.calls = 0
        def action(self, observation, action):
            self.calls += 1
            raise TimeoutError('uncertain')
    client = Client()
    with pytest.raises(TimeoutError):
        floor_limited_action(client, {'joints_deg': [0] * 6, 'gripper_mm': 90}, {'joints_deg': [1] * 6, 'gripper_mm': 0})
    assert client.calls == 1


def test_feedback_teacher_preserves_relative_grasp_and_levels_cube(observation):
    from scipy.spatial.transform import Rotation
    client = FakeSimulator()
    task = {'cube_xy_mm': [350, 0], 'cube_size_mm': 50, 'cube_yaw_deg': 0, 'lift_clearance_mm': 100, 'hold_seconds': 5}
    teacher = Teacher(client, task)
    observation['cube_pose']['quaternion_xyzw'] = Rotation.from_euler('x', 8, degrees=True).as_quat().tolist()
    teacher._feedback_lift_target(observation, 115)
    desired = client.calls[-1][1]['pose']
    def matrix(p):
        value = np.eye(4)
        value[:3, :3] = Rotation.from_quat(p['quaternion_xyzw']).as_matrix()
        value[:3, 3] = p['position_mm']
        return value
    expected_cube = matrix(desired) @ np.linalg.inv(matrix(observation['grasp_pose'])) @ matrix(observation['cube_pose'])
    np.testing.assert_allclose(expected_cube[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(expected_cube[:3, 3], [350, 0, 139], atol=1e-6)


def test_locked_console_prevents_native_launch(monkeypatch):
    import pytest
    import fly_brain.simulator as module
    monkeypatch.setattr(module, 'screen_locked', lambda: True)
    with pytest.raises(RuntimeError, match='Unlock the Mac'):
        with module.Simulator(binary='/does/not/exist'):
            raise AssertionError('Must not launch while locked')


def test_lock_parser_uses_current_user_session(monkeypatch):
    import os
    import fly_brain.simulator as module
    text = '"IOConsoleUsers" = ({"kCGSSessionUserIDKey"=' + str(os.getuid()) + ',"CGSSessionScreenIsLocked"=Yes})\n'
    monkeypatch.setattr(module.sys, 'platform', 'darwin')
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **kw: text)
    assert module.screen_locked()


def test_native_floor_limiting_is_recorded_without_a_retry():
    from fly_brain.simulator import ReBotClient, native_floor_adjustment
    client = object.__new__(ReBotClient)
    client.session = {"token": "owned"}; client.action_id = 4
    calls = []
    proposed = {"joints_deg": [0, -179, -95, 10, 0, 90], "gripper_mm": 90}
    accepted = [0, -133, -95, 10, 0, 90]
    observation = {"episode_id": "same", "frame_id": 70}
    def call(name, **arguments):
        calls.append((name, arguments))
        return {"accepted": True, "floor_limited": True, "target_joints_deg": accepted,
                "target_gripper_mm": 90, "observation": observation}
    client.call = call
    assert client.action(observation, proposed) == observation
    assert len(calls) == 1 and client.action_id == 5
    report = native_floor_adjustment(client, proposed)
    assert report["accepted_target"]["joints_deg"] == accepted
    assert report["proposed_action"] == proposed
    assert client.session == {"token": "owned"}
