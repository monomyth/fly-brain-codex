from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse


@pytest.fixture
def graph():
    # Two sensory cells, a relay, and two distinct output cells.
    matrix = sparse.csr_matrix(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                        [.7, .3, 0, 0, 0], [.5, 0, .5, 0, 0], [0, .2, .8, 0, 0]], dtype=np.float32))
    return SimpleNamespace(matrix=matrix, inputs=np.array([0, 1]), outputs=np.array([3, 4]),
                           features=np.eye(5, dtype=np.float32), ids=np.arange(100, 105), manifest={"graph_id": "a" * 64})


@pytest.fixture
def observation():
    return {"episode_id": "example", "input_mode": "state", "simulation_time": .5, "phase": "ready",
            "joints_deg": [0, -100, -90, 20, 0, 0], "joint_velocity_deg_s": [0] * 6,
            "gripper_mm": 50, "gripper_velocity_mm_s": 0,
            "tool_pose": {"position_mm": [350, 0, 100], "quaternion_xyzw": [0, .70710678, 0, .70710678]},
            "grasp_pose": {"position_mm": [350, 0, 120], "quaternion_xyzw": [0, .70710678, 0, .70710678]},
            "cube_pose": {"position_mm": [350, 0, 24], "quaternion_xyzw": [0, 0, 0, 1]},
            "finger_contacts": {"left": False, "right": False},
            "goal": {"clearance_mm": 100, "tilt_tolerance_deg": 5, "hold_seconds": 5}}
