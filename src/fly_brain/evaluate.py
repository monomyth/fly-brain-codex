"""Autonomous evaluation, with hold and release scored outside policy input."""
import math
from pathlib import Path
import time
import numpy as np
from .assets import write_json
from .dataset import EpisodeWriter
from .activity import ActivityPublisher
from .simulator import configure_episode, floor_limited_action, release_for_cleanup, current_episode_task, native_floor_adjustment


def wilson(successes, total):
    if total == 0:
        return [0., 1.]
    z = 1.959963984540054
    fraction = successes / total
    denominator = 1 + z * z / total
    center = (fraction + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total * total)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def released(evaluation):
    return (not evaluation.get("held") and "floor" in evaluation.get("contacts", [])
            and evaluation.get("clearance_mm", 999) < 2 and evaluation.get("linear_speed_mm_s", 999) < 5)


def evaluate(client, policy, tasks, output, release=True, release_timeout=10, progress=print, record_images=False, floor_projection=False, current_episode=False, expected_episode_id=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for number, spec in enumerate(tasks):
        task_override = spec.get("task", spec)
        verified = bool(spec.get("teacher_verified", False))
        held_out = verified and spec.get("split") == "test" and spec.get("configuration_key") not in policy.metadata.get("training_configuration_keys", [])
        result = {"success": False, "hold_success": False, "release_success": False,
                  "floor_projection": floor_projection, "floor_adjustments": 0,
                  "teacher_verified_configuration": verified, "held_out_configuration": held_out, "task": task_override, "ablation": policy.ablation}
        writer = None
        activity = None
        latencies = []
        try:
            task = current_episode_task(client, policy.mode, expected_episode_id) if current_episode else configure_episode(client, task_override)
            if current_episode:
                result["task"] = task
            first = client.observe(images=policy.mode == "vision" or record_images)
            policy.reset(first); policy.act(first); policy.reset(first)
            provenance = "malecns" if policy.metadata["architecture"]["kind"] == "malecns" else "conventional"
            writer = EpisodeWriter(output, task, provenance, policy.hz, {"checkpoint": policy.metadata["name"], "ablation": policy.ablation})
            activity = ActivityPublisher.for_policy(policy, client)
            client.call("rebot_experiment_control", action="start")
            client.connect(provenance=provenance, model_id=policy.metadata["name"][:190])
            hold_time = None
            deadline = time.monotonic() + task["timeout_seconds"] + release_timeout + 10
            while time.monotonic() < deadline:
                began = time.monotonic()
                observation = client.observe(images=policy.mode == "vision" or record_images)
                evaluation = client.call("rebot_get_evaluation")
                if evaluation.get("success") and hold_time is None:
                    hold_time = observation["simulation_time"]
                    result["hold_success"] = True
                    result["hold_evaluation"] = evaluation
                if hold_time is not None:
                    if not release:
                        result["success"] = True
                        break
                    if released(evaluation):
                        result.update(success=True, release_success=True, release_evaluation=evaluation)
                        break
                    if observation["simulation_time"] - hold_time > release_timeout:
                        raise TimeoutError("Hold passed but the neural policy did not release within the deadline")
                if observation["phase"] not in ("running", "completed"):
                    raise RuntimeError(f"Episode ended: {observation['phase']}")
                # The policy never receives evaluation, task overrides, or a teacher action.
                action = policy.act(observation)
                projection = None
                if floor_projection:
                    _, action, projection = floor_limited_action(client, observation, action)
                else:
                    client.action(observation, action)
                native_projection = native_floor_adjustment(client, action)
                if native_projection:
                    projection = {**(projection or {}), **native_projection}
                result["floor_adjustments"] += int(projection is not None)
                if activity is not None:
                    activity.publish(policy.state, observation)
                latency = 1000 * (time.monotonic() - began)
                latencies.append(latency)
                writer.append(observation, action, evaluation=evaluation, latency_ms=latency, intervention=projection)
                time.sleep(max(0, 1 / policy.hz - (time.monotonic() - began)))
            else:
                raise TimeoutError("Autonomous episode exceeded its wall-clock deadline")
        except KeyboardInterrupt:
            result.update(error="Run interrupted", error_type="KeyboardInterrupt", interrupted=True)
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            result["error"] = str(error)
            result["error_type"] = type(error).__name__
        finally:
            cleanup_error = release_for_cleanup(client)
            if activity is not None:
                activity.close()
                if activity.error:
                    result["activity_error"] = activity.error
            if cleanup_error:
                result["cleanup_error"] = cleanup_error
            if latencies:
                result["p95_loop_ms"] = float(np.percentile(latencies, 95))
                result["max_loop_ms"] = max(latencies)
            if writer is not None:
                result["episode_directory"] = str(writer.finish(result, complete=not result.get("interrupted", False)))
            reports.append(result)
            verified_reports = [r for r in reports if r["teacher_verified_configuration"] and r["held_out_configuration"]]
            successes = sum(r["success"] for r in reports)
            report = {"checkpoint": policy.metadata["name"], "ablation": policy.ablation,
                      "observation_mode": policy.mode, "requires_release": release, "floor_projection": floor_projection,
                      "attempts": len(reports), "successes": successes,
                      "hold_successes": sum(r["hold_success"] for r in reports),
                      "success_rate": successes / len(reports), "wilson_95": wilson(successes, len(reports)),
                      "teacher_verified_attempts": len(verified_reports),
                      "teacher_verified_successes": sum(r["success"] for r in verified_reports),
                      "promotion_target_met": len(verified_reports) >= 100 and sum(r["success"] for r in verified_reports) / len(verified_reports) >= .9,
                      "repeated_training_seeds_verified": False, "episodes": reports}
            write_json(output / "evaluation.json", report)
            progress(f"Autonomous episode {number + 1}: {'success' if result['success'] else result.get('error', 'failed')}", flush=True)
        if result.get("interrupted"):
            break
    return report
