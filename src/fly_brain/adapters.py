"""Explicit policy inputs and bounded joint/gripper action semantics."""
import base64
import io
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

JOINT_LOWER = np.rad2deg([-2.8, -3.14, -3.14, -1.87, -1.57, -3.14])
JOINT_UPPER = np.rad2deg([2.8, 0, 0, 1.57, 1.57, 3.14])
BASE_FEATURES = ([f"joint_{i}" for i in range(6)] + [f"joint_velocity_{i}" for i in range(6)]
                 + ["gripper", "gripper_velocity"] + [f"tool_xyz_{i}" for i in range(3)]
                 + [f"tool_rotation_{i}" for i in range(6)] + [f"grasp_xyz_{i}" for i in range(3)]
                 + ["left_contact", "right_contact", "goal_clearance", "goal_tilt", "goal_hold", "dt"])
STATE_FEATURES = [f"cube_relative_{i}" for i in range(3)] + [f"cube_rotation_{i}" for i in range(6)] + ["initial_cube_side", "cube_visible"]
VISION_FEATURES = [f"{camera}_{name}" for camera in ("Front", "Top")
                   for name in ("cx", "cy", "width", "height", "area", "visible", "confidence")]


def feature_schema(mode):
    if mode not in ("state", "vision"):
        raise ValueError("Observation mode must be state or vision")
    return {"version": 1, "mode": mode, "names": BASE_FEATURES + (STATE_FEATURES if mode == "state" else VISION_FEATURES),
            "privileged_state": mode == "state", "excluded": ["phase", "owner", "episode_id", "frame_id", "evaluator", "expert_action", "task_time"]}


def rotation6(pose):
    quaternion = np.asarray(pose["quaternion_xyzw"], dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-8:
        raise ValueError("Invalid pose quaternion")
    return Rotation.from_quat(quaternion).as_matrix()[:, :2].ravel()


def decode_image(camera, base_directory=None):
    if "jpeg_base64" in camera:
        payload = io.BytesIO(base64.b64decode(camera["jpeg_base64"], validate=True))
    elif "file" in camera and base_directory is not None:
        base = Path(base_directory).resolve()
        payload = (base / camera["file"]).resolve()
        if not payload.is_relative_to(base):
            raise ValueError("Image path escapes episode directory")
    else:
        raise ValueError("Missing camera image")
    with Image.open(payload) as image:
        if image.width * image.height > 4_000_000:
            raise ValueError("Image exceeds supported pixel budget")
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def orange_features(rgb):
    """Image-only baseline: orange pixels, distinct from the yellow robot parts."""
    from scipy.ndimage import label, find_objects
    pixels = rgb.astype(np.float32) / 255
    r, g, b = pixels.transpose(2, 0, 1)
    mask = (r > .38) & (g > .08) & (g < .72 * r) & (b < .55 * g) & (r - b > .28)
    labels, count = label(mask)
    if count == 0:
        return np.zeros(7, dtype=np.float32)
    areas = np.bincount(labels.ravel()); areas[0] = 0
    component = int(areas.argmax())
    if areas[component] < 4:
        return np.zeros(7, dtype=np.float32)
    yy, xx = find_objects(labels)[component - 1]
    height, width = mask.shape
    return np.array([(xx.start + xx.stop) / width - 1, (yy.start + yy.stop) / height - 1,
                     (xx.stop - xx.start) / width, (yy.stop - yy.start) / height,
                     areas[component] / (width * height), 1.0,
                     min(1, areas[component] / max(1, (xx.stop - xx.start) * (yy.stop - yy.start)))], dtype=np.float32)


class ObservationAdapter:
    def __init__(self, mode="state", hz=5):
        self.schema = feature_schema(mode)
        self.mode = mode
        self.hz = hz
        self.last_time = None
        self.last_frame = None
        self.episode = None
        self.cube_side = None

    def reset(self, observation):
        if observation.get("input_mode") != self.mode:
            raise ValueError(f"Policy requires {self.mode} observations")
        self.episode = observation["episode_id"]
        self.last_time = None
        self.last_frame = None
        if self.mode == "state":
            # Reset observations must come from the settled floor placement.
            z = observation["cube_pose"]["position_mm"][2]
            self.cube_side = float(np.clip(2 * (z + 1), 10, 90))

    def encode(self, observation, base_directory=None):
        if observation["episode_id"] != self.episode:
            raise ValueError("Episode changed without resetting neural state")
        if observation.get("input_mode") != self.mode:
            raise ValueError("Observation mode changed")
        frame = observation.get("frame_id")
        if frame is not None and self.last_frame is not None and frame <= self.last_frame:
            raise ValueError("Simulation frame did not advance; check for a paused or locked display before resuming")
        self.last_frame = frame
        now = float(observation["simulation_time"])
        dt = 1 / self.hz if self.last_time is None else now - self.last_time
        if dt < 0 or not np.isfinite(dt):
            raise ValueError("Nonmonotonic simulation time")
        self.last_time = now
        tool = observation["tool_pose"]
        grasp = observation["grasp_pose"]
        goal = observation["goal"]
        features = np.concatenate([
            np.asarray(observation["joints_deg"]) / 180,
            np.asarray(observation["joint_velocity_deg_s"]) / 30,
            [observation["gripper_mm"] / 90, observation["gripper_velocity_mm_s"] / 25],
            np.asarray(tool["position_mm"]) / 500, rotation6(tool),
            np.asarray(grasp["position_mm"]) / 500,
            [observation["finger_contacts"]["left"], observation["finger_contacts"]["right"],
             goal["clearance_mm"] / 100, goal["tilt_tolerance_deg"] / 5, goal["hold_seconds"] / 5, min(dt, 2)],
        ])
        if self.mode == "state":
            cube = observation["cube_pose"]
            extra = np.concatenate([(np.asarray(cube["position_mm"]) - grasp["position_mm"]) / 500,
                                    rotation6(cube), [self.cube_side / 90, 1]])
        else:
            # No lookup of cube_pose, setup state, or evaluator fields in this branch.
            cameras = {camera["name"]: camera for camera in observation.get("images", [])}
            from .cameras import LEGACY_RIG
            if any(camera.get("rig_revision",LEGACY_RIG)!=LEGACY_RIG for camera in cameras.values()):
                raise ValueError("This legacy vision adapter requires the old Front/Top rig; use the Front/Gripper retinal controller")
            if not all(name in cameras for name in ("Front", "Top")):
                raise ValueError("Vision policy requires synchronized Front and Top images")
            extra = np.concatenate([orange_features(decode_image(cameras[name], base_directory)) for name in ("Front", "Top")])
        encoded = np.concatenate([features, extra]).astype(np.float32)
        if encoded.shape != (len(self.schema["names"]),) or not np.isfinite(encoded).all():
            raise ValueError("Invalid observation features")
        return encoded


class ActionAdapter:
    def __init__(self, hz=5, max_degrees_per_second=30):
        if not 1 <= hz <= 30:
            raise ValueError("Control frequency must be 1–30 Hz")
        if not np.isfinite(max_degrees_per_second) or max_degrees_per_second <= 0:
            raise ValueError("Positive finite action speed required")
        self.hz = float(hz)
        self.step_degrees = min(30, max_degrees_per_second) / hz

    def label(self, observation, joints, grip):
        delta = np.asarray(joints, dtype=np.float64) - observation["joints_deg"]
        # One fraction preserves the path direction for the coordinated native controller.
        delta *= min(1.0, self.step_degrees / max(np.max(np.abs(delta)), 1e-9))
        if grip not in (0, 90):
            raise ValueError("Teacher labels must preserve explicit open/close intent")
        return np.r_[delta / self.step_degrees, float(grip == 90)].astype(np.float32)

    def decode(self, observation, normalized_delta, open_probability):
        delta = np.asarray(normalized_delta, dtype=np.float64)
        if delta.shape != (6,) or not np.isfinite(delta).all() or not np.isfinite(open_probability):
            raise ValueError("Non-finite or malformed neural action")
        joints = np.clip(np.asarray(observation["joints_deg"]) + np.clip(delta, -1, 1) * self.step_degrees,
                         JOINT_LOWER, JOINT_UPPER)
        return {"joints_deg": joints.tolist(), "gripper_mm": 90.0 if open_probability >= 0.5 else 0.0}
