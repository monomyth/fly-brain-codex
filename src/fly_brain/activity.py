"""Optional, lossy display telemetry; never part of the robot control protocol."""
import base64
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import warnings

import numpy as np
import pyarrow.parquet as pq
from .assets import home, locked, write_json, checked_files, sha256_file, ATTRIBUTION
from .connectome import resolve_graph


def prepare_layout(root=None, graph_id=None):
    """Join by body ID, retaining only finite soma coordinates. Never invent locations."""
    root = home(root)
    graph, manifest = resolve_graph(root, graph_id, verify=False)
    graph_id = manifest["graph_id"]
    directory = root / "visualizations" / graph_id / "soma-v1"
    with locked(root / ".locks" / f"soma-{graph_id}"):
        if (directory / "layout.json").exists():
            layout = json.loads((directory / "layout.json").read_text())
            checked_files(directory, layout)
            return directory, layout
        ids = np.load(graph / "node-ids.npy", allow_pickle=False)
        table = pq.read_table(graph / "neuron-features.parquet", columns=["bodyId", "somaLocation"]).to_pydict()
        lookup = dict(zip(table["bodyId"], table["somaLocation"]))
        indices, positions = [], []
        for i, body_id in enumerate(ids):
            position = lookup.get(int(body_id))
            if position is not None and len(position) == 3 and np.isfinite(position).all():
                indices.append(i)
                positions.append(position)
        if not positions:
            raise ValueError("The graph has no finite soma positions to display")
        xyz = np.asarray(positions, dtype=np.float64)
        center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
        scale = float(np.ptp(xyz, axis=0).max() / 2) or 1.0
        # Fixed display space only: EM x -> right, EM z -> down, EM y -> depth.
        vertices = ((xyz - center) / scale)[:, [0, 2, 1]]
        vertices[:, 1] *= -1
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "positions.f32").write_bytes(vertices.astype("<f4").tobytes())
        np.save(directory / "indices.npy", np.asarray(indices, dtype=np.int64), allow_pickle=False)
        np.save(directory / "body-ids.npy", ids[indices], allow_pickle=False)
        (directory / "ATTRIBUTION.md").write_text(ATTRIBUTION + "\nDisplay: finite soma positions joined by body ID; centered and uniformly scaled. No missing locations are reconstructed.\n")
        layout = {"schema": "malecns-soma-layout-v1", "graph_id": graph_id,
                  "count": len(indices), "total_neurons": len(ids), "missing_positions": len(ids) - len(indices),
                  "source_center": center.tolist(), "source_scale": scale,
                  "display_axes": ["x", "-z", "y"], "positions_format": "little-endian float32 xyz",
                  "files": {name: sha256_file(directory / name) for name in
                            ("positions.f32", "indices.npy", "body-ids.npy", "ATTRIBUTION.md")}}
        write_json(directory / "layout.json", layout)
    return directory, layout


def activity_bytes(state, indices, total_neurons):
    """Mean absolute continuous state across channels, on a fixed [0, 1] scale."""
    if hasattr(state, "detach"):
        state = state.detach().cpu().numpy()
    state = np.asarray(state)
    if state.ndim != 3 or state.shape[0] != 1 or state.shape[1] != total_neurons or state.shape[2] < 1:
        raise ValueError("Activity shape does not match the selected graph")
    activity = np.abs(state[0, indices, :]).mean(axis=-1)
    if not np.isfinite(activity).all():
        raise ValueError("Nonfinite neuron activity")
    return np.rint(np.clip(activity, 0, 1) * 255).astype(np.uint8).tobytes()


class ActivityPublisher:
    """One pending frame, background atomic writes, no feedback into inference."""
    def __init__(self, control_directory, layout_directory, layout):
        self.path = Path(control_directory) / "brain-activity.json"
        self.layout_directory = Path(layout_directory)
        self.layout = layout
        self.indices = np.load(self.layout_directory / "indices.npy", allow_pickle=False)
        self._condition = threading.Condition()
        self._pending = None
        self._closing = False
        self.error = None
        self._thread = threading.Thread(target=self._worker, name="brain-display", daemon=True)
        self._thread.start()

    @classmethod
    def for_policy(cls, policy, client):
        if policy.metadata["architecture"]["kind"] not in ("malecns", "malecns_visual_dopamine", "malecns_motor_dopamine") or not hasattr(client, "directory") or not hasattr(policy, "state"):
            return None
        try:
            directory, layout = prepare_layout(getattr(policy, "root", None), policy.metadata["graph_id"])
            publisher = cls(client.directory, directory, layout)
            publisher.policy = policy
            return publisher
        except Exception as error:
            warnings.warn(f"Brain overlay unavailable: {error}", RuntimeWarning)
            return None

    def publish(self, state, observation):
        if self.error or self._closing:
            return
        try:
            values = activity_bytes(state, self.indices, self.layout["total_neurons"])
            frame = {"schema": "malecns-activity-v1", "episode_id": observation["episode_id"],
                     "frame_id": observation["frame_id"], "simulation_time": observation["simulation_time"],
                     "generated_at": time.time(), "layout_directory": str(self.layout_directory),
                     "graph_id": self.layout["graph_id"], "count": self.layout["count"],
                     "metric": "mean_absolute_state", "values": base64.b64encode(values).decode("ascii")}
            policy = getattr(self, "policy", None)
            if policy is not None and hasattr(policy, "activity_metadata"):
                frame.update(policy.activity_metadata())
            with self._condition:
                self._pending = frame
                self._condition.notify()
        except Exception as error:
            self._failed(error)

    def _failed(self, error):
        self.error = str(error)
        warnings.warn(f"Brain overlay disabled: {error}", RuntimeWarning)

    def _worker(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closing)
                if self._pending is None:
                    return
                frame, self._pending = self._pending, None
            temporary = None
            try:
                fd, temporary = tempfile.mkstemp(prefix=".brain-activity-", dir=self.path.parent)
                with os.fdopen(fd, "w") as stream:
                    json.dump(frame, stream, separators=(",", ":"), allow_nan=False)
                os.replace(temporary, self.path)
            except Exception as error:
                self._failed(error)
                return
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)

    def close(self):
        with self._condition:
            self._closing = True
            self._condition.notify()
        self._thread.join(timeout=1)
