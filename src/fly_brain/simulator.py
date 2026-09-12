"""Same-user local ReBot IPC and an optional isolated native simulation process."""
import ctypes
import json
import os
import re
import sys
from pathlib import Path
import socket
import stat
import subprocess
import tempfile
import time

DEFAULT_BINARY = "~/github/rebot-motion-lab-codex/dist/ReBot Motion Lab Codex.app/Contents/MacOS/ReBotMotionLab"


def screen_locked():
    if sys.platform != "darwin":
        return False
    try:
        data = subprocess.check_output(["/usr/sbin/ioreg", "-w0", "-n", "Root", "-d1"], timeout=5, text=True, stderr=subprocess.DEVNULL)
        if '"IOConsoleUsers" =' in data:
            sessions = data.split('"IOConsoleUsers" =', 1)[1].split("\n", 1)[0]
            for entry in re.findall(r"\{[^{}]*\}", sessions):
                if re.search(r'"kCGSSessionUserIDKey"\s*=\s*' + str(os.getuid()) + r'(?:,|\})', entry):
                    return bool(re.search(r'"CGSSessionScreenIsLocked"\s*=\s*Yes', entry))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return False


class SimulatorError(RuntimeError):
    pass


class ReBotClient:
    def __init__(self, directory):
        self.directory = Path(directory)
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("Control directory must be private, owned by this user, and not a symlink")
        self.session = None
        self.action_id = 0

    def call(self, name, **arguments):
        request = json.dumps({"tool": name, "arguments": arguments}, allow_nan=False).encode() + b"\n"
        if len(request) > 1_048_576:
            raise ValueError("Request exceeds IPC limit")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(5)
            sock.connect(str(self.directory / "control.sock"))
            uid, gid = ctypes.c_uint(), ctypes.c_uint()
            libc = ctypes.CDLL(None)
            if libc.getpeereid(sock.fileno(), ctypes.byref(uid), ctypes.byref(gid)) or uid.value != os.getuid():
                raise PermissionError("Unexpected simulator peer")
            sock.sendall(request)
            response = bytearray()
            while b"\n" not in response:
                chunk = sock.recv(8192)
                if not chunk:
                    raise ConnectionError("Simulator disconnected; motion is not automatically retried")
                response.extend(chunk)
                if len(response) > 1_048_577:
                    raise ValueError("Response exceeds IPC limit")
        result = json.loads(response.split(b"\n", 1)[0])
        if not result.get("ok"):
            raise SimulatorError(result.get("error", "Simulator rejected request"))
        return result["data"]

    def observe(self, images=False):
        return self.call("rebot_get_observation", images=images)

    def connect(self, provenance="conventional", model_id="fly-brain-teacher-v1"):
        self.session = self.call("rebot_controller_connect", provenance=provenance, model_id=model_id)
        self.action_id = 0
        return self.session["observation"]

    def action(self, observation, action):
        if self.session is None:
            raise ValueError("Connect before submitting actions")
        result = self.call("rebot_controller_action", token=self.session["token"],
                           episode_id=observation["episode_id"], observed_frame_id=observation["frame_id"],
                           action_id=self.action_id, **action)
        self.action_id += 1
        self.last_action_result = result
        return result["observation"]

    def release(self):
        if self.session is not None:
            try:
                self.call("rebot_experiment_control", action="takeover")
            finally:
                self.session = None

    def wait_ready(self, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.call("rebot_get_evaluation")
            if result["phase"] == "ready":
                return result
            if result["phase"] in ("invalid", "physics_error"):
                raise SimulatorError(result.get("error", result["phase"]))
            time.sleep(.05)
        raise TimeoutError("Simulator did not settle: " + json.dumps(result, sort_keys=True))


class Simulator:
    def __init__(self, directory=None, binary=DEFAULT_BINARY, log=None):
        self.directory = Path(directory).expanduser() if directory else None
        self.binary = Path(binary).expanduser()
        self.process = None
        self.log_path = Path(log) if log else None
        self.log = None
        self.power_guard = None

    def __enter__(self):
        if screen_locked():
            raise RuntimeError("Unlock the Mac before starting live simulation; RealityKit cannot advance reliably while the screen is locked")
        if self.directory is None:
            if not self.binary.is_file():
                raise FileNotFoundError(self.binary)
            self.directory = Path(tempfile.mkdtemp(prefix="fly-brain-sim-", dir="/tmp"))
            self.log = (self.log_path or self.directory / "native.log").open("w")
            self.process = subprocess.Popen([str(self.binary), "--external-controller"],
                env={**os.environ, "REBOT_CONTROL_DIRECTORY": str(self.directory), "REBOT_MCP_NO_LAUNCH": "1"},
                stdout=self.log, stderr=self.log, start_new_session=True)
        self.client = ReBotClient(self.directory)
        try:
            if sys.platform == "darwin":
                command = ["/usr/bin/caffeinate", "-di"]
                if self.process:
                    command += ["-w", str(self.process.pid)]
                self.power_guard = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                if self.process and self.process.poll() is not None:
                    raise RuntimeError("Isolated simulator exited; inspect native.log")
                try:
                    state = self.client.call("rebot_get_state")
                    if state.get("scene_ready"):
                        if state.get("hardware_connected") is not False:
                            raise RuntimeError("This pipeline only supports the simulator")
                        from .assets import sha256_file
                        self.client.provenance = {"transport": "same-user Unix socket", "isolated": self.process is not None, "app_version": state.get("app_version"),
                                                  "binary_sha256": sha256_file(self.binary) if self.process else None}
                        if self.process and len(self.binary.parents) > 4:
                            repository = self.binary.parents[4]
                            if (repository / ".git").exists():
                                import hashlib
                                self.client.provenance["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
                                diff = subprocess.check_output(["git", "diff", "HEAD", "--", "Sources"], cwd=repository)
                                self.client.provenance["source_diff_sha256"] = hashlib.sha256(diff).hexdigest()
                                from .assets import canonical_hash
                                self.client.provenance["source_tree_sha256"] = canonical_hash({
                                    str(path.relative_to(repository)): sha256_file(path)
                                    for path in sorted((repository / "Sources").rglob("*.swift"))})
                        return self.client
                except (OSError, SimulatorError):
                    pass
                time.sleep(.2)
            raise TimeoutError("Native simulator did not initialize")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        try:
            if hasattr(self, "client"):
                self.cleanup_error = release_for_cleanup(self.client)
        finally:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait(timeout=5)
            if self.power_guard and self.power_guard.poll() is None:
                self.power_guard.terminate()
                self.power_guard.wait(timeout=5)
            if self.log:
                self.log.close()


def configure_episode(client, task):
    if isinstance(client, EpisodeSimulator):
        client = client.start_episode()
    client.release()
    placement_keys = {"cube_xy_mm", "cube_size_mm", "cube_yaw_deg", "seed"}
    fixed = {key: value for key, value in task.items() if key not in placement_keys}
    previous = getattr(client, "_fixed_task", None)
    if previous == fixed and task.get("placement_jitter_mm", 0) == 0:
        # Keep the robot colliders and attached camera rigs. Reset restores the
        # starting arm pose; placement changes only the cube and episode state.
        client.call("rebot_reset_episode", **({"seed": task["seed"]} if "seed" in task else {}))
        client.wait_ready()
        xy = task.get("cube_xy_mm", [350, 0])
        client.call("rebot_place_cube", x_mm=xy[0], y_mm=xy[1],
                    size_mm=task.get("cube_size_mm", 50), yaw_deg=task.get("cube_yaw_deg", 0))
    else:
        client.call("rebot_configure_task", task=task)
    client.wait_ready()
    client._fixed_task = fixed
    return client.call("rebot_get_task")["task"]


def native_floor_adjustment(client, proposed):
    """Keep native contact limiting visible in episode records without retrying it."""
    result = getattr(client, "last_action_result", None)
    if result and result.get("floor_limited"):
        return {"native_floor_limited": True, "proposed_action": proposed,
                "accepted_target": {"joints_deg": result["target_joints_deg"],
                                    "gripper_mm": result["target_gripper_mm"]}}
    return None


def floor_limited_action(client, observation, proposed, max_halvings=8):
    """Project a proposed action toward the current pose using the native floor check.

    No cube pose, task goal, teacher, or IK target is used. Only a definitive
    endpoint-floor rejection is handled; uncertain responses always propagate.
    The simulator validates the swept path and applies its usual rate limits.
    """
    current = observation['joints_deg']
    aperture = observation['gripper_mm']
    for attempt in range(max_halvings + 2):
        fraction = 2.0 ** -attempt if attempt <= max_halvings else 0.0
        action = {'joints_deg': [a + fraction * (b - a) for a, b in zip(current, proposed['joints_deg'])],
                  'gripper_mm': aperture + fraction * (proposed['gripper_mm'] - aperture)}
        try:
            accepted = client.action(observation, action)
            return accepted, action, {'fraction': fraction, 'rejected_candidates': attempt,
                                      'proposed_action': proposed} if attempt else None
        except SimulatorError as error:
            if 'endpoint intersects the floor' not in str(error):
                raise
    raise SimulatorError('Even a stationary action failed the native floor check')


class EpisodeSimulator:
    """A fresh owned native process per learned-policy trial.

    Recreating the physics world avoids carrying contact/CCD state out of extreme
    failed-policy poses. The neural model stays loaded in the caller process.
    """
    def __init__(self, binary=DEFAULT_BINARY, log=None):
        self.binary = binary
        self.log_path = Path(log) if log else None
        self._simulator = None
        self._client = None
        self._episode_number = 0

    def __enter__(self):
        return self

    def start_episode(self):
        self.__exit__(None, None, None)
        self._episode_number += 1
        log = self.log_path.with_name(f'{self.log_path.stem}-{self._episode_number:03d}{self.log_path.suffix}') if self.log_path else None
        self._simulator = Simulator(binary=self.binary, log=log)
        self._client = self._simulator.__enter__()
        return self._client

    def release(self):
        if self._client is not None:
            self._client.release()

    def __getattr__(self, name):
        if self._client is None:
            raise AttributeError(name)
        return getattr(self._client, name)

    def __exit__(self, *args):
        if self._simulator is not None:
            self._simulator.__exit__(*args)
        self._simulator = None
        self._client = None


def release_for_cleanup(client):
    """Report cleanup failures without replacing the primary run error."""
    try:
        client.release()
    except Exception as error:
        return {"type": type(error).__name__, "message": str(error)}
    return None


def current_episode_task(client, mode, expected_episode_id=None):
    """Read the placed cube without resetting, reconfiguring, or taking over."""
    info = client.call("rebot_get_task")
    state, task = info["state"], info["task"]
    if expected_episode_id is not None and state["episode_id"] != expected_episode_id:
        raise ValueError("The cube episode changed while the model was loading; start again")
    if state["phase"] != "ready":
        raise ValueError("Apply the cube or reset to a ready episode before running the model")
    if task["input_mode"] != mode:
        raise ValueError(f"This checkpoint needs {mode} observations; select the matching input mode")
    return task
