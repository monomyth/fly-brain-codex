import base64
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fly_brain.activity import prepare_layout, activity_bytes, ActivityPublisher


def fixture_graph(tmp_path):
    graph_id = "a" * 64
    graph = tmp_path / "prepared" / graph_id
    graph.mkdir(parents=True)
    (graph / "manifest.json").write_text(json.dumps({"graph_id": graph_id}))
    np.save(graph / "node-ids.npy", np.array([40, 10, 30, 20, 50]))
    # Deliberately shuffled, missing and nonfinite annotations catch index joins.
    pq.write_table(pa.table({"bodyId": [10, 20, 30, 40, 50], "somaLocation":
        [[1., 2., 3.], None, [4., 8., 12.], [7., 9., 15.], [float("nan"), 0., 0.]]}), graph / "neuron-features.parquet")
    return graph_id


def test_layout_keeps_body_id_alignment_and_never_invents_missing_positions(tmp_path):
    graph_id = fixture_graph(tmp_path)
    directory, layout = prepare_layout(tmp_path, graph_id)
    assert np.load(directory / "body-ids.npy").tolist() == [40, 10, 30]
    assert np.load(directory / "indices.npy").tolist() == [0, 1, 2]
    assert layout["missing_positions"] == 2
    vertices = np.fromfile(directory / "positions.f32", dtype="<f4").reshape(-1, 3)
    assert np.isfinite(vertices).all() and abs(vertices).max() <= 1
    before = (directory / "positions.f32").stat().st_mtime_ns
    assert prepare_layout(tmp_path, graph_id)[1] == layout
    assert (directory / "positions.f32").stat().st_mtime_ns == before
    (directory / "positions.f32").write_bytes(b"corrupt")
    with pytest.raises(ValueError): prepare_layout(tmp_path, graph_id)


def test_activity_preserves_neuron_mapping_and_uses_absolute_channel_mean():
    state = np.array([[[-1., .5], [0., 0.], [.25, -.25]]])
    assert list(activity_bytes(state, np.array([2, 0, 1]), 3)) == [64, 191, 0]
    with pytest.raises(ValueError): activity_bytes(state, np.array([0]), 4)
    state[0, 0, 0] = float("nan")
    with pytest.raises(ValueError): activity_bytes(state, np.array([0]), 3)


def test_latest_snapshot_has_episode_frame_and_real_values(tmp_path):
    graph_id = fixture_graph(tmp_path)
    directory, layout = prepare_layout(tmp_path, graph_id)
    control = tmp_path / "control"; control.mkdir()
    publisher = ActivityPublisher(control, directory, layout)
    state = np.zeros((1, 5, 2)); state[0, 1] = [-.4, .8]
    observation = {"episode_id": "episode-b", "frame_id": 73, "simulation_time": 12.5}
    publisher.publish(state, observation)
    publisher.close()
    frame = json.loads((control / "brain-activity.json").read_text())
    assert frame["episode_id"] == "episode-b" and frame["frame_id"] == 73
    assert frame["simulation_time"] == 12.5 and frame["graph_id"] == graph_id
    assert list(base64.b64decode(frame["values"])) == [0, 153, 0]
    assert not list(control.glob(".brain-activity-*"))


def test_display_io_failure_does_not_propagate_into_control(tmp_path):
    directory, layout = prepare_layout(tmp_path, fixture_graph(tmp_path))
    publisher = ActivityPublisher(tmp_path / "absent-directory", directory, layout)
    with pytest.warns(RuntimeWarning, match="overlay disabled"):
        publisher.publish(np.zeros((1, 5, 1)), {"episode_id": "x", "frame_id": 1, "simulation_time": 1.})
        publisher.close()
    assert publisher.error
