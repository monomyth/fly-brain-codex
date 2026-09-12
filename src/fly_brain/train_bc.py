"""Sequence-aware imitation training and a cached frozen-reservoir diagnostic."""
import collections
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .adapters import feature_schema
from .assets import canonical_hash, sha256_file, write_json, locked
from .checkpoint import finish_checkpoint, load_checkpoint, save_weights
from .connectome import Graph, resolve_graph
from .dataset import load_split, normalization
from .neural_core import FlyController, RecurrentBaseline


def imitation_loss(predicted, target, weights=None):
    joint = (predicted[..., :6].tanh() - target[..., :6]).square().mean(dim=-1)
    grip = F.binary_cross_entropy_with_logits(predicted[..., 6], target[..., 6], reduction="none")
    loss = 4 * joint + grip
    return (loss * weights).mean() if weights is not None else loss.mean()


def prepare(episodes, norm, stage_weights):
    mean, std = np.array(norm["mean"]), np.array(norm["std"])
    for episode in episodes:
        episode["x"] = torch.tensor((episode["features"] - mean) / std, dtype=torch.float32)
        episode["y"] = torch.from_numpy(episode["labels"])
        episode["weight"] = torch.tensor([stage_weights.get(stage, 1.) for stage in episode["stages"]], dtype=torch.float32)


def sequence_pass(model, episodes, optimizer=None, sequence_length=32, shuffle=False):
    total, count = 0., 0
    order = torch.randperm(len(episodes)).tolist() if shuffle else range(len(episodes))
    model.train(optimizer is not None)
    for index in order:
        episode = episodes[index]
        state = model.initial_state()
        for start in range(0, len(episode["x"]), sequence_length):
            stop = min(len(episode["x"]), start + sequence_length)
            predictions = []
            with torch.set_grad_enabled(optimizer is not None):
                for observation in episode["x"][start:stop]:
                    output, state = model.step(observation.unsqueeze(0), state)
                    predictions.append(output.squeeze(0))
                loss = imitation_loss(torch.stack(predictions), episode["y"][start:stop], episode["weight"][start:stop])
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
            state = state.detach()
            total += loss.item() * (stop - start); count += stop - start
        if optimizer is not None and len(episodes) >= 10:
            processed = list(order).index(index) + 1
            if processed % 5 == 0:
                print(json.dumps({"training_episodes_processed": processed, "total_episodes": len(episodes), "running_loss": total / max(1, count)}), flush=True)
    return total / max(1, count)


def cache_reservoir(model, episodes, cache_root=None, graph_id=None):
    import hashlib
    parameters = {name: hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                  for name, tensor in model.state_dict().items() if not name.startswith(("head.", "read_norm."))}
    core_id = canonical_hash({"version": 1, "graph_id": graph_id, "config": model.config, "parameters": parameters})
    stats = {"hits": 0, "built": 0}
    for episode in episodes:
        input_id = hashlib.sha256(episode["x"].numpy().tobytes()).hexdigest()
        cached = Path(cache_root) / core_id / (input_id + ".npy") if cache_root else None
        def compute():
            state = model.initial_state()
            outputs = []
            with torch.no_grad():
                for features in episode["x"]:
                    state = model.advance_state(features.unsqueeze(0), state)
                    outputs.append(model.readout(state).squeeze(0).clone())
            return torch.stack(outputs)
        if cached is None:
            episode["reservoir"] = compute(); stats["built"] += 1
            continue
        cached.parent.mkdir(parents=True, exist_ok=True)
        with locked(cached.with_suffix(".lock")):
            metadata = cached.with_suffix(".json")
            if cached.exists() and metadata.exists():
                record = json.loads(metadata.read_text())
                if sha256_file(cached) != record["sha256"]:
                    raise ValueError("Reservoir feature cache checksum mismatch")
                array = np.load(cached, allow_pickle=False)
                if array.shape != (len(episode["x"]), len(model.outputs) * model.channels) or not np.isfinite(array).all():
                    raise ValueError("Malformed reservoir feature cache")
                episode["reservoir"] = torch.from_numpy(array)
                stats["hits"] += 1
            else:
                episode["reservoir"] = compute()
                partial = cached.with_suffix(".partial")
                with partial.open("wb") as stream:
                    np.save(stream, episode["reservoir"].numpy(), allow_pickle=False)
                os.replace(partial, cached)
                write_json(metadata, {"sha256": sha256_file(cached), "core_id": core_id, "input_id": input_id})
                stats["built"] += 1
    return stats


def reservoir_pass(model, episodes, optimizer=None):
    features = torch.cat([episode["reservoir"] for episode in episodes])
    targets = torch.cat([episode["y"] for episode in episodes])
    weights = torch.cat([episode["weight"] for episode in episodes])
    order = torch.randperm(len(features)) if optimizer else torch.arange(len(features))
    total = 0.
    for indices in order.split(256):
        with torch.set_grad_enabled(optimizer is not None):
            output = model.head(model.read_norm(features[indices]))
            loss = imitation_loss(output, targets[indices], weights[indices])
            if optimizer:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
        total += loss.item() * len(indices)
    return total / len(features)


def train(args, root):
    if args.epochs < 1 or args.sequence_length < 1 or args.learning_rate <= 0:
        raise ValueError("Training epochs, sequence length, and learning rate must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Training creates a new checkpoint directory; choose another output")
    working = output.with_name(output.name + ".partial")
    working.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(args.seed); np.random.seed(args.seed)
    split = json.loads((Path(args.dataset) / "splits.json").read_text())
    training = load_split(args.dataset, "train", args.mode, split)
    validation = load_split(args.dataset, "validation", args.mode, split)
    if not training or not validation:
        raise ValueError("Nonempty whole-episode train and validation splits are required")
    hz_set = {e["manifest"]["hz"] for e in training + validation}
    if len(hz_set) != 1:
        raise ValueError("Mixed collection rates require explicit resampling")
    norm = normalization(training)
    frequencies = collections.Counter(stage for e in training for stage in e["stages"])
    weights = {stage: sum(frequencies.values()) / (len(frequencies) * n) for stage, n in frequencies.items()}
    prepare(training, norm, weights); prepare(validation, norm, weights)
    graph = None
    if args.kind == "malecns":
        graph = Graph(resolve_graph(root, args.graph_id)[0], verify=False)
        model = FlyController(graph, len(norm["mean"]), channels=args.channels, updates=args.updates, seed=args.seed)
    else:
        model = RecurrentBaseline(len(norm["mean"]), hidden_dim=530, seed=args.seed)
    if args.resume:
        model, previous = load_checkpoint(args.resume, root)
        if previous["observation_schema"] != feature_schema(args.mode):
            raise ValueError("Cannot resume a different observation schema")
        if args.kind != previous["architecture"]["kind"]:
            raise ValueError("Cannot resume a different model kind")
        if graph is not None and graph.manifest["graph_id"] != previous["graph_id"]:
            raise ValueError("Cannot resume on a different anatomical graph")
        norm = previous["normalization"]
        prepare(training, norm, weights); prepare(validation, norm, weights)
    started = time.monotonic()
    cache_started = started
    if args.reservoir:
        if args.kind != "malecns":
            raise ValueError("Reservoir diagnostic requires the MaleCNS architecture")
        model.freeze_reservoir()
        print("Caching full-graph reservoir states for train and validation episodes", flush=True)
        cache_stats = cache_reservoir(model, training + validation, root / "feature-cache", graph.manifest["graph_id"])
        print(json.dumps({"reservoir_cache": cache_stats}), flush=True)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    cache_seconds = time.monotonic() - cache_started
    if getattr(args, "resume_optimizer", False):
        if not args.resume:
            raise ValueError("--resume-optimizer requires --resume")
        optimizer.load_state_dict(torch.load(Path(args.resume) / "optimizer.pt", map_location="cpu", weights_only=True))
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
    best = float("inf"); history = []
    for epoch in range(args.epochs):
        began = time.monotonic()
        if args.reservoir:
            train_loss = reservoir_pass(model, training, optimizer)
            val_loss = reservoir_pass(model, validation)
        else:
            train_loss = sequence_pass(model, training, optimizer, args.sequence_length, shuffle=True)
            val_loss = sequence_pass(model, validation, sequence_length=args.sequence_length)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss,
               "seconds": time.monotonic() - began}
        history.append(row); write_json(working / "training.json", history)
        print(json.dumps(row), flush=True)
        if val_loss < best:
            best = val_loss
            save_weights(model, working)
            torch.save(optimizer.state_dict(), working / "optimizer.pt")
    provenance = {e["directory"].name: sha256_file(e["directory"] / "steps.jsonl") for e in training + validation}
    if args.mode == "vision":
        for episode in training + validation:
            image_hashes = {p.name: sha256_file(p) for p in sorted((episode["directory"] / "images").glob("*.jpg"))}
            provenance[episode["directory"].name] = canonical_hash([provenance[episode["directory"].name], image_hashes])
    metadata = {"name": output.name, "architecture": model.config,
                "graph_id": None if graph is None else graph.manifest["graph_id"],
                "training_method": "frozen_reservoir_readout" if args.reservoir else "behaviour_cloning",
                "observation_schema": feature_schema(args.mode), "normalization": norm,
                "action_schema": {"version": 1, "joints": "six bounded deltas; common teacher scaling", "gripper": "binary requested open/close intent"},
                "hz": next(iter(hz_set)), "seed": args.seed, "split_id": split["split_id"],
                "dataset_id": canonical_hash(provenance), "episode_hashes": provenance,
                "training_episodes": len(training), "validation_episodes": len(validation),
                "training_domain": {"cube_sizes_mm": sorted({e["manifest"]["task"].get("cube_size_mm", 50) for e in training}),
                                    "hold_seconds": sorted({e["manifest"]["task"].get("hold_seconds", 5) for e in training}),
                                    "clearance_mm": sorted({e["manifest"]["task"].get("lift_clearance_mm", 100) for e in training})},
                "training_configuration_keys": sorted({e["manifest"]["configuration_key"] for e in training}),
                "episode_simulator_builds": {e["directory"].name: e["manifest"].get("simulator") for e in training + validation},
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "best_validation_loss": best, "training_seconds": time.monotonic() - started, "reservoir_cache_seconds": cache_seconds,
                "closed_loop_validated": False, "promotion_target": "90/100 held-out admissible successes across repeated seeds",
                "resume_from": str(args.resume) if args.resume else None,
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "package_source_hash": canonical_hash({p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")}),
                "simulator_provenance": json.loads((Path(args.dataset) / "simulator-provenance.json").read_text()) if (Path(args.dataset) / "simulator-provenance.json").exists() else None}
    finish_checkpoint(working, metadata)
    os.rename(working, output)
    return {"checkpoint": str(output), **metadata}
