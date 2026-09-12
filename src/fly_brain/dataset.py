"""Durable aligned episode records and whole-configuration dataset splits."""
import base64
import hashlib
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
from .adapters import ObservationAdapter
from .cameras import observation_contract
from .assets import canonical_hash, write_json


def configuration_key(task):
    keys = ("cube_xy_mm", "cube_size_mm", "cube_yaw_deg", "initial_joints_deg", "initial_gripper_mm")
    return canonical_hash({key: task.get(key) for key in keys})


class EpisodeWriter:
    def __init__(self, dataset, task, provenance, hz, metadata=None):
        self.directory = Path(dataset) / ("episode-" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.stream = (self.directory / "steps.jsonl.partial").open("w")
        self.manifest = {"schema": "fly-brain-episode-v1", "task": task, "provenance": provenance,
                         "hz": hz, "configuration_key": configuration_key(task), "steps": 0,
                         "complete": False, "success": False, "created_unix": time.time(), **(metadata or {})}
        write_json(self.directory / "manifest.json", self.manifest)

    def append(self, observation, action, label=None, stage=None, evaluation=None, latency_ms=None, intervention=None, expert_target=None, expert_path=None):
        observation = deepcopy(observation)
        step = self.manifest["steps"]
        if observation.get("images"):
            names, revision = observation_contract(observation)
            contract = {"camera_names":list(names), "camera_rig_revision":revision}
            for key, value in contract.items():
                if key in self.manifest and self.manifest[key] != value:
                    raise ValueError("Camera rig changed within an episode")
            self.manifest.update(contract)
        for camera in observation.get("images", []):
            name = camera["name"]
            if name not in ("Front", "Top", "Gripper"):
                raise ValueError("Unexpected camera name")
            relative = f"images/{step:05d}-{name}.jpg"
            target = self.directory / relative
            target.parent.mkdir(exist_ok=True)
            payload = base64.b64decode(camera.pop("jpeg_base64"), validate=True)
            target.write_bytes(payload)
            camera["sha256"] = hashlib.sha256(payload).hexdigest()
            camera["file"] = relative
        row = {"observation": observation, "action": action, "expert_label": None if label is None else np.asarray(label).tolist(),
               "stage": stage, "evaluation": evaluation, "loop_latency_ms": latency_ms, "intervention": intervention,
               "expert_target": deepcopy(expert_target), "expert_path": deepcopy(expert_path)}
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()
        self.manifest["steps"] += 1

    def finish(self, result, complete=True):
        self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close()
        os.replace(self.directory / "steps.jsonl.partial", self.directory / "steps.jsonl")
        self.manifest.update(complete=complete, success=bool(result.get("success")), result=result)
        write_json(self.directory / "manifest.json", self.manifest)
        return self.directory


def split_dataset(dataset, seed=0, include_dagger_failures=False):
    dataset = Path(dataset)
    episodes = []
    for path in sorted(dataset.glob("episode-*/manifest.json")):
        manifest = json.loads(path.read_text())
        eligible = manifest["success"] or (include_dagger_failures and manifest["provenance"] == "teacher_assisted" and manifest.get("teacher_version") in ("cube_feedback_v2","cube_feedback_v3_synchronized","cube_feedback_v4_approach_close"))
        if manifest["complete"] and eligible and manifest["steps"] > 0 and manifest.get("result", {}).get("cadence_passed", True):
            episodes.append((path.parent.name, manifest["configuration_key"]))
    groups = sorted({key for _, key in episodes})
    if len(groups) < 3:
        raise ValueError("At least three distinct configurations are required for train/validation/test splitting")
    existing = dataset / "splits.json"
    if existing.exists():
        previous = json.loads(existing.read_text())
        if previous["seed"] != seed:
            raise ValueError("Existing split seed cannot change: preserve held-out configurations")
        group_split = dict(previous["configuration_split"])
        for key in groups:
            if key not in group_split:
                bucket = int(canonical_hash([seed, key])[:8], 16) % 10
                group_split[key] = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
    else:
        rng = np.random.default_rng(seed); rng.shuffle(groups)
        validation = max(1, round(len(groups) * .1)); test = max(1, round(len(groups) * .1))
        group_split = {key: "test" if i < test else "validation" if i < test + validation else "train" for i, key in enumerate(groups)}
    splits = {name: [episode for episode, key in episodes if group_split[key] == name] for name in ("train", "validation", "test")}
    report = {"schema": "fly-brain-splits-v1", "seed": seed, "unit": "whole_configuration",
              "splits": splits, "configuration_split": group_split, "include_dagger_failures": include_dagger_failures}
    report["split_id"] = canonical_hash(report)
    write_json(dataset / "splits.json", report)
    return report


def read_episode(directory, mode="state"):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Incomplete episodes cannot be used for training")
    rows = [json.loads(line) for line in (directory / "steps.jsonl").open()]
    adapter = ObservationAdapter(mode, manifest["hz"])
    observations = []
    labels, stages = [], []
    excluded_suffix = 0
    for index, row in enumerate(rows):
        evaluation = row.get("evaluation") or {}
        if (manifest["provenance"] == "teacher_assisted" and evaluation.get("clearance_mm", 999) < 2
                and evaluation.get("tilt_deg", 0) > 30 and not evaluation.get("held")):
            excluded_suffix = len(rows) - index
            break
        observation = deepcopy(row["observation"])
        if mode == "vision":
            observation["input_mode"] = "vision"
            observation.pop("cube_pose", None)
        if not observations:
            adapter.reset(observation)
        observations.append(adapter.encode(observation, directory))
        if row["expert_label"] is None:
            raise ValueError("Missing teacher action label")
        labels.append(row["expert_label"]); stages.append(row["stage"])
    if not observations:
        raise ValueError("No valid teacher-labelled prefix remains")
    manifest = {**manifest, "training_excluded_overturned_suffix": excluded_suffix}
    return {"features": np.array(observations, dtype=np.float32), "labels": np.array(labels, dtype=np.float32),
            "stages": stages, "manifest": manifest, "directory": directory}


def load_split(dataset, split, mode="state", specification=None):
    dataset = Path(dataset)
    specification = specification or json.loads((dataset / "splits.json").read_text())
    result = []
    for name in specification["splits"][split]:
        path = (dataset / name).resolve()
        if not path.is_relative_to(dataset.resolve()):
            raise ValueError("Split path escapes dataset")
        result.append(read_episode(path, mode))
    return result


def normalization(episodes):
    # Caller must pass only the train split.
    features = np.concatenate([episode["features"] for episode in episodes])
    return {"mean": features.mean(axis=0).tolist(), "std": np.maximum(features.std(axis=0), .05).tolist(),
            "fit_split": "train", "samples": len(features)}


def export_tasks(dataset, split="test"):
    dataset = Path(dataset)
    specification = json.loads((dataset / "splits.json").read_text())
    tasks = {}
    for name in specification["splits"][split]:
        manifest = json.loads((dataset / name / "manifest.json").read_text())
        if not manifest["complete"] or not manifest["success"]:
            raise ValueError("Only completed successful teacher episodes qualify")
        tasks[manifest["configuration_key"]] = {"task": manifest["task"], "teacher_verified": True,
             "configuration_key": manifest["configuration_key"], "source_episode": name,
             "split": split, "split_id": specification["split_id"]}
    return list(tasks.values())
