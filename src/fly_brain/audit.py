"""Validate durable episode contents before fitting a model."""
import json
from pathlib import Path
import numpy as np
from .adapters import ActionAdapter, decode_image
from .assets import sha256_file, write_json
from .cameras import observation_contract, LEGACY_CAMERAS, LEGACY_RIG


def audit_dataset(dataset, output=None):
    dataset = Path(dataset)
    report = {"schema": "fly-brain-dataset-audit-v1", "episodes": [], "errors": [], "complete_steps": 0}
    for file in sorted(dataset.glob("episode-*/manifest.json")):
        directory = file.parent
        manifest = json.loads(file.read_text())
        if not manifest["complete"]:
            report["episodes"].append({"episode": directory.name, "complete": False})
            continue
        try:
            rows = [json.loads(line) for line in (directory / "steps.jsonl").open()]
            if len(rows) != manifest["steps"]:
                raise ValueError("Manifest and row counts disagree")
            last_frame = -1; last_time = -1.; episode_id = None
            image_count = 0
            for row in rows:
                observation = row["observation"]
                if episode_id is None:
                    episode_id = observation["episode_id"]
                if observation["episode_id"] != episode_id:
                    raise ValueError("Episode IDs were mixed")
                frame = observation["frame_id"]; now = observation["simulation_time"]
                if frame <= last_frame or now <= last_time:
                    raise ValueError("Non-increasing frame IDs or simulation times")
                last_frame = frame; last_time = now
                label = row["expert_label"]
                if label is not None:
                    label = np.asarray(label)
                    if label.shape != (7,) or not np.isfinite(label).all() or np.any(np.abs(label[:6]) > 1.000001) or label[6] not in (0, 1):
                        raise ValueError("Invalid teacher action label")
                    if manifest["provenance"] == "conventional":
                        adapter=ActionAdapter(manifest["hz"])
                        absolute=manifest.get("execution_mode")=="absolute_targets"
                        if absolute:
                            target=row.get("expert_target")
                            if not target:raise ValueError("Absolute-target demonstration lacks its expert goal")
                            expected_label=adapter.label(observation,target["joints_deg"],target["gripper_mm"])
                            if not np.allclose(label,expected_label,atol=1e-5):
                                raise ValueError("Teacher label does not match its expert goal")
                            expected=target
                        else:expected=adapter.decode(observation,label[:6],label[6])
                        # Recorded exploration/recovery may deliberately execute a different goal.
                        # Supervision must still match the expert goal, not that intervention.
                        if not row.get("intervention") and (not np.allclose(expected["joints_deg"],row["action"]["joints_deg"],atol=1e-5) or expected["gripper_mm"]!=row["action"]["gripper_mm"]):
                            raise ValueError("Teacher label does not match its requested action")
                cameras = observation.get("images", [])
                if manifest.get("images_each_decision") and not cameras:
                    raise ValueError("Missing synchronized camera pair")
                if cameras:
                    names, revision = observation_contract(observation)
                    expected = (tuple(manifest.get("camera_names", LEGACY_CAMERAS)), manifest.get("camera_rig_revision", LEGACY_RIG))
                    if (names, revision) != expected:
                        raise ValueError("Recorded cameras disagree with episode manifest")
                if cameras and observation.get("image_state_skew_seconds") != 0:
                    raise ValueError("Camera/telemetry skew was not zero")
                for camera in cameras:
                    image = decode_image(camera, directory)
                    if image.shape[:2] != (camera["height"], camera["width"]):
                        raise ValueError("Camera image dimensions disagree with calibration")
                    if camera.get("sha256") and sha256_file(directory / camera["file"]) != camera["sha256"]:
                        raise ValueError("Image checksum mismatch")
                    image_count += 1
            report["complete_steps"] += len(rows)
            report["episodes"].append({"episode": directory.name, "complete": True, "steps": len(rows), "images": image_count,
                                       "teacher_success": manifest["success"], "cadence_passed": manifest.get("result", {}).get("cadence_passed")})
        except (KeyError, ValueError, OSError) as error:
            report["errors"].append({"episode": directory.name, "error": str(error)})
    report["valid"] = not report["errors"] and bool(report["episodes"])
    if output is not None:
        write_json(output, report)
    return report
