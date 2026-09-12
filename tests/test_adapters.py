import base64
from copy import deepcopy
import io
import numpy as np
from PIL import Image
import pytest
from fly_brain.adapters import ActionAdapter, ObservationAdapter, feature_schema, orange_features
from fly_brain.teacher import grasp_pose


def camera():
    pixels = np.zeros((40, 60, 3), dtype=np.uint8)
    pixels[10:25, 20:35] = [255, 100, 5]
    stream = io.BytesIO(); Image.fromarray(pixels).save(stream, format="JPEG")
    return {"jpeg_base64": base64.b64encode(stream.getvalue()).decode()}


def test_close_intent_is_not_measured_aperture(observation):
    adapter = ActionAdapter(5)
    label = adapter.label(observation, observation["joints_deg"], 0)
    assert label[-1] == 0
    assert adapter.decode(observation, label[:6], label[-1])["gripper_mm"] == 0


def test_coordinated_step_and_joint_bounds(observation):
    adapter = ActionAdapter(5)
    target = np.array(observation["joints_deg"]) + [12, -6, 0, 0, 0, 0]
    label = adapter.label(observation, target, 90)
    np.testing.assert_allclose(label[:2], [1, -.5])
    action = adapter.decode(observation, label[:6], 1)
    np.testing.assert_allclose(np.array(action["joints_deg"]) - observation["joints_deg"], [6, -3, 0, 0, 0, 0])


def test_schema_ignores_evaluator_and_episode_metadata(observation):
    first = ObservationAdapter(); first.reset(observation); expected = first.encode(observation)
    changed = deepcopy(observation)
    changed.update(phase="completed", owner="teacher", frame_id=9999, task_time=9999,
                   evaluator={"success": True}, expert_action=[999] * 7)
    second = ObservationAdapter(); second.reset(changed)
    np.testing.assert_array_equal(expected, second.encode(changed))


def test_vision_features_cannot_read_exact_cube_pose(observation):
    obs = deepcopy(observation); obs["input_mode"] = "vision"
    obs["images"] = [{"name": name, **camera()} for name in ("Front", "Top")]
    adapter = ObservationAdapter("vision"); adapter.reset(obs); first = adapter.encode(obs)
    obs["cube_pose"] = {"malformed": "not read"}
    other = ObservationAdapter("vision"); other.reset(obs); second = other.encode(obs)
    np.testing.assert_array_equal(first, second)
    assert len(first) == len(feature_schema("vision")["names"])


def test_reset_and_nonfinite_rejection(observation):
    adapter = ObservationAdapter(); adapter.reset(observation)
    changed = deepcopy(observation); changed["episode_id"] = "new"
    with pytest.raises(ValueError): adapter.encode(changed)
    observation["joints_deg"][0] = float("nan")
    with pytest.raises(ValueError): adapter.encode(observation)


def test_small_cube_teacher_tip_stays_above_floor():
    for size in [10, 20, 50, 90]:
        target = grasp_pose([350, 0], size, 0)
        assert target["position_mm"][2] - 20 >= 1


def test_absent_detection_has_explicit_mask():
    assert not orange_features(np.zeros((30, 30, 3), dtype=np.uint8)).any()


def test_stalled_physics_frame_is_not_advanced_as_a_new_neural_step(observation):
    observation['frame_id'] = 4
    adapter = ObservationAdapter(); adapter.reset(observation); adapter.encode(observation)
    with pytest.raises(ValueError, match='frame did not advance'):
        adapter.encode(observation)
