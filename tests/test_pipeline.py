from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy import sparse
from fly_brain.adapters import feature_schema
from fly_brain.assets import Asset, verify_asset, sha256_file, artifact_manifest, write_json
from fly_brain.checkpoint import save_weights, finish_checkpoint, load_checkpoint, export_checkpoint, export_core, apply_core
from fly_brain.dataset import EpisodeWriter, split_dataset, load_split, normalization, export_tasks
from fly_brain.neural_core import FlyController
from fly_brain.policy import Policy
from fly_brain.train_bc import train
from fly_brain.train_rl import gae, action_log_probability, ppo_update
from fly_brain.evaluate import wilson, released


def install_graph(root, graph):
    directory = root / "prepared" / graph.manifest["graph_id"]
    directory.mkdir(parents=True)
    sparse.save_npz(directory / "graph-csr.npz", graph.matrix)
    np.save(directory / "node-ids.npy", graph.ids)
    np.save(directory / "neuron-features.npy", graph.features)
    write_json(directory / "io-populations.json", {"inputs": graph.inputs.tolist(), "outputs": graph.outputs.tolist()})
    artifact_manifest(directory, {"schema": "malecns-graph-v1", "graph_id": graph.manifest["graph_id"]})
    write_json(root / "prepared/latest.json", {"graph_id": graph.manifest["graph_id"]})


def test_checksum_validation_does_not_overwrite(tmp_path):
    file = tmp_path / "test.feather"; file.write_bytes(b"correct")
    asset = Asset("test.feather", "remote.feather", 7, sha256_file(file), "1")
    assert verify_asset(file, asset)["sha256"] == asset.sha256
    file.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_asset(file, asset)
    assert file.read_bytes() == b"corrupt"


def make_dataset(path, observation):
    for i in range(6):
        task = {"cube_xy_mm": [330 + i * 5, 0], "cube_size_mm": 50, "cube_yaw_deg": 0}
        writer = EpisodeWriter(path, task, "conventional", 5)
        for t in range(8):
            obs = deepcopy(observation); obs["episode_id"] = str(i); obs["simulation_time"] += t * .2
            obs["cube_pose"]["position_mm"][0] = task["cube_xy_mm"][0]
            writer.append(obs, {"joints_deg": obs["joints_deg"], "gripper_mm": 0}, [0] * 7, "hold")
        writer.finish({"success": True})
    return split_dataset(path)


def test_episode_splits_and_train_only_normalization(tmp_path, observation):
    splits = make_dataset(tmp_path, observation)
    assert len({ep for eps in splits["splits"].values() for ep in eps}) == 6
    for a in splits["splits"]:
        for b in splits["splits"]:
            if a != b:
                assert not set(splits["splits"][a]) & set(splits["splits"][b])
    episodes = load_split(tmp_path, "train")
    norm = normalization(episodes)
    assert norm["samples"] == 8 * len(episodes)
    assert norm["fit_split"] == "train"
    assert all(task["teacher_verified"] for task in export_tasks(tmp_path))


def test_train_checkpoint_and_load_in_second_project(tmp_path, observation, graph):
    shared = tmp_path / "shared"; install_graph(shared, graph)
    dataset = tmp_path / "episodes"; make_dataset(dataset, observation)
    args = Namespace(output=shared / "checkpoints/test", dataset=dataset, mode="state", kind="malecns", epochs=2,
                     sequence_length=4, learning_rate=.001, seed=4, reservoir=False, resume=None,
                     graph_id=None, channels=1, updates=2)
    report = train(args, shared)
    assert np.isfinite(report["best_validation_loss"])
    checkpoint = args.output
    model, manifest = load_checkpoint(checkpoint, shared)
    assert all("operator" not in key for key in model.state_dict())
    policy = Policy(checkpoint, shared); policy.reset(observation); result = policy.act(observation)
    assert len(result["joints_deg"]) == 6 and result["gripper_mm"] in (0, 90)
    # Same graph root, another project: loading does not copy or download graph assets.
    other = tmp_path / "other-project"; other.mkdir()
    second = Policy(checkpoint, shared); second.reset(observation)
    assert second.act(observation) == result
    exported = tmp_path / "export"
    export_checkpoint(checkpoint, exported, shared, include_graph=True)
    third = Policy(exported / "checkpoint", exported); third.reset(observation)
    assert third.act(observation) == result
    assert not (exported / "checkpoint/optimizer.pt").exists()
    with pytest.raises(FileExistsError):
        export_checkpoint(checkpoint, exported, shared)
    core = tmp_path / "core"
    export_core(checkpoint, core)
    new_robot = FlyController(graph, input_dim=12)
    sensory = new_robot.sensory.weight.detach().clone()
    apply_core(new_robot, core, graph.manifest["graph_id"])
    torch.testing.assert_close(new_robot.sensory.weight, sensory)
    torch.testing.assert_close(new_robot.response.weight, model.response.weight)
    with pytest.raises(ValueError):
        apply_core(new_robot, core, "b" * 64)


def test_gae_terminal_and_truncation_bootstrap():
    rewards = torch.tensor([1., 2.]); values = torch.tensor([.5, .8])
    advantage, target = gae(rewards, values, torch.tensor([0., 1.]), bootstrap=999, gamma=1, lam=1)
    torch.testing.assert_close(target, torch.tensor([3., 2.]))
    _, target = gae(rewards, values, torch.zeros(2), bootstrap=4, gamma=1, lam=1)
    torch.testing.assert_close(target, torch.tensor([7., 6.]))


def test_recurrent_ppo_update_has_finite_gradients(graph):
    model = FlyController(graph, 4)
    critic = torch.nn.Linear(7, 1)
    log_std = torch.nn.Parameter(torch.full((6,), -2.))
    features = torch.randn(6, 4)
    outputs = []; state = None
    with torch.no_grad():
        for x in features:
            out, state = model.step(x[None], state); outputs.append(out.squeeze(0))
        out = torch.stack(outputs); latent = out[:, :6].clone(); opened = torch.zeros(6)
        old = action_log_probability(out, latent, opened, log_std)
        values = critic(out).squeeze(-1)
    rollout = dict(features=features, latent=latent, opened=opened, log_probabilities=old,
                   values=values, rewards=torch.ones(6), terminals=torch.tensor([0, 0, 0, 0, 0, 1.]), bootstrap=0.)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(critic.parameters()) + [log_std], lr=.001)
    result = ppo_update(model, critic, log_std, rollout, optimizer, chunk_length=3)
    assert np.isfinite(result["ppo_loss"])


def test_release_scoring_and_small_sample_uncertainty():
    assert released(dict(held=False, contacts=["floor"], clearance_mm=.1, linear_speed_mm_s=0))
    assert not released(dict(held=True, contacts=["floor"], clearance_mm=.1, linear_speed_mm_s=0))
    assert wilson(3, 3)[0] < .5
    assert wilson(90, 100)[1] < 1
