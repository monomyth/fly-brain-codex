"""Privileged conventional teacher; never used inside autonomous policy inference."""
import math
import numpy as np
from .adapters import ActionAdapter


def grasp_pose(xy, size, yaw, clearance=0):
    # Keep the tool tip >= 1 mm above the -1 mm floor for small cubes.
    grasp_height = max(size / 2, 22.0)
    half = math.radians(yaw) / 2
    h = math.sqrt(.5)
    return {"position_mm": [float(xy[0]), float(xy[1]), -1 + grasp_height + clearance],
            "quaternion_xyzw": [-math.sin(half) * h, math.cos(half) * h, math.sin(half) * h, math.cos(half) * h]}


class Teacher:
    version = "cube_feedback_v4_approach_close"
    stages = ("approach", "pregrasp", "descend", "close", "lift_low", "lift_middle", "lift", "hold", "release", "done")

    def __init__(self, client, task, hz=5, release=True):
        self.client = client
        self.task = task
        self.adapter = ActionAdapter(hz)
        self.release_required = release
        self.stage = "approach"
        self.stage_started = None
        self.hold_joints = None
        self.hold_succeeded = False
        self.recovery_count = 0
        self.closing_started = False
        self.path = {}
        size = task["cube_size_mm"]
        approach_clearance = min(160.0, 185.0 - max(size / 2, 22.0))
        for stage, clearance in [("approach", approach_clearance), ("pregrasp", 20), ("descend", 0),
                                 ("lift_low", 20), ("lift_middle", 60), ("lift", task["lift_clearance_mm"] + 15)]:
            self.path[stage] = client.call("rebot_solve_pose", pose=grasp_pose(task["cube_xy_mm"], size, task["cube_yaw_deg"], clearance),
                                           frame="grasp", gripper_mm=90 if stage in ("approach", "pregrasp", "descend") else size)["joints_deg"]

    def _feedback_lift_target(self, observation, clearance, cube_pose=None):
        from scipy.spatial.transform import Rotation
        def matrix(pose):
            value = np.eye(4)
            value[:3, :3] = Rotation.from_quat(pose["quaternion_xyzw"]).as_matrix()
            value[:3, 3] = pose["position_mm"]
            return value
        cube = matrix(observation["cube_pose"] if cube_pose is None else cube_pose)
        grasp = matrix(observation["grasp_pose"])
        desired_cube = np.eye(4)
        desired_cube[:3, :3] = Rotation.from_euler("z", self.task["cube_yaw_deg"], degrees=True).as_matrix()
        desired_cube[:3, 3] = [cube[0, 3], cube[1, 3], -1 + self.task["cube_size_mm"] / 2 + clearance]
        # Estimate the current relative grasp from synchronized measurements.
        # This commands the arm; the cube stays a free contact-physics body.
        desired_grasp = desired_cube @ np.linalg.inv(cube) @ grasp
        pose = {"position_mm": desired_grasp[:3, 3].tolist(),
                "quaternion_xyzw": Rotation.from_matrix(desired_grasp[:3, :3]).as_quat().tolist()}
        return self.client.call("rebot_solve_pose", pose=pose, frame="grasp", gripper_mm=observation["gripper_mm"])["joints_deg"]

    def _advance(self, observation):
        self.stage = self.stages[self.stages.index(self.stage) + 1]
        self.stage_started = observation["simulation_time"]

    def act(self, observation, evaluation):
        now = observation["simulation_time"]
        if self.stage_started is None:
            self.stage_started = now
        if now - self.stage_started > (self.task["hold_seconds"] + 15 if self.stage == "hold" else 20):
            raise TimeoutError(f"Teacher stuck at {self.stage}")
        # Recover a cube displaced on the floor during policy-driven DAgger.
        cube_pose = evaluation.get("cube_pose")
        on_floor = evaluation.get("clearance_mm", 999) < 2
        if self.stage not in ("release","done") and on_floor and evaluation.get("tilt_deg", 0) > 30 and not any(observation["finger_contacts"].values()):
            raise RuntimeError("Cube overturned outside the teacher recovery domain")
        if (cube_pose is not None and on_floor and not evaluation.get("success")
                and not any(observation["finger_contacts"].values())):
            xy = cube_pose["position_mm"][:2]
            displaced = np.linalg.norm(np.asarray(xy) - self.task["cube_xy_mm"]) > 5
            lost_during_lift = self.stage in ("lift_low", "lift_middle", "lift", "hold")
            if displaced or lost_during_lift:
                if self.recovery_count >= 3:
                    raise RuntimeError("Teacher recovery limit reached")
                from scipy.spatial.transform import Rotation
                revised = {**self.task, "cube_xy_mm": xy,
                           "cube_yaw_deg": float(Rotation.from_quat(cube_pose["quaternion_xyzw"]).as_euler("xyz", degrees=True)[2])}
                next_teacher = Teacher(self.client, revised, self.adapter.hz, self.release_required)
                self.path = next_teacher.path; self.task = revised
                self.stage = "approach"; self.stage_started = now; self.recovery_count += 1; self.closing_started=False
        # During transport, preserve the measured grasp-to-cube transform and
        # correct the cube's levelness, including imperfect policy grasps.
        lift_clearances = {"lift_low": 20, "lift_middle": 60, "lift": self.task["lift_clearance_mm"] + 15, "hold": self.task["lift_clearance_mm"] + 15}
        if self.stage in lift_clearances and all(observation["finger_contacts"].values()) and ("cube_pose" in observation or cube_pose is not None):
            target = self._feedback_lift_target(observation, lift_clearances[self.stage],observation.get("cube_pose",cube_pose))
            self.path["lift" if self.stage == "hold" else self.stage] = target
        # Recompute the bounded correction from the actual visited joint state.
        if self.stage in self.path:
            target = self.path[self.stage]
            grip = 90 if self.stage in ("approach", "pregrasp", "descend") else 0
            if self.stage=='descend':
                from scipy.spatial.transform import Rotation
                desired=grasp_pose(self.task['cube_xy_mm'],self.task['cube_size_mm'],self.task['cube_yaw_deg'])
                distance=np.linalg.norm(np.asarray(observation['grasp_pose']['position_mm'])-desired['position_mm'])
                angle=(Rotation.from_quat(desired['quaternion_xyzw']).inv()*Rotation.from_quat(observation['grasp_pose']['quaternion_xyzw'])).magnitude()
                self.closing_started=self.closing_started or (distance<12 and angle<.15)
                if self.closing_started:grip=0
            reached = max(abs(a - b) for a, b in zip(observation["joints_deg"], target)) < .2
            if reached and (grip == 0 or observation["gripper_mm"] > 89):
                self.hold_joints = list(observation["joints_deg"])
                self._advance(observation)
                return self.act(observation, evaluation)
        elif self.stage == "close":
            target, grip = self.hold_joints, 0
            if all(observation["finger_contacts"].values()):
                self._advance(observation)
                return self.act(observation, evaluation)
        elif self.stage == "hold":
            target, grip = self.path["lift"], 0
            if evaluation.get("success"):
                self.hold_succeeded = True
                self.hold_joints = list(observation["joints_deg"])
                self._advance(observation)
                if not self.release_required:
                    self.stage = "done"
                return self.act(observation, evaluation)
        elif self.stage == "release":
            target, grip = self.hold_joints, 90
            if (evaluation.get("clearance_mm", 999) < 2 and not evaluation.get("held")
                    and "floor" in evaluation.get("contacts", []) and evaluation.get("linear_speed_mm_s", 999) < 5
                    and observation["gripper_mm"] > 89):
                self._advance(observation)
        else:
            target, grip = self.hold_joints, 90 if self.release_required else 0
        self.last_target = {"joints_deg": list(target), "gripper_mm": float(grip)}
        self.last_execution_target=dict(self.last_target)
        label = self.adapter.label(observation, target, grip)
        action = self.adapter.decode(observation, label[:6], label[6])
        return action, label


def placement_tasks(count, seed=0, sizes=(50,), xy_radius=25, yaw_range=5):
    if count < 1 or any(size < 10 or size > 90 for size in sizes):
        raise ValueError("Positive episode count and cube sizes between 10 and 90 mm required")
    rng = np.random.default_rng(seed)
    for i in range(count):
        yield {"cube_xy_mm": [float(350 + rng.uniform(-xy_radius, xy_radius)), float(rng.uniform(-xy_radius, xy_radius))],
               "cube_size_mm": float(sizes[i % len(sizes)]), "cube_yaw_deg": float(rng.uniform(-yaw_range, yaw_range)),
               "input_mode": "state", "hold_seconds": 5, "timeout_seconds": 90, "seed": seed + i}
