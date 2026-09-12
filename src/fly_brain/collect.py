"""Synchronized teacher and DAgger collection at the policy decision rate."""
import json
from pathlib import Path
import time
import numpy as np
from .assets import write_json,sha256_file
from .dataset import EpisodeWriter
from .simulator import configure_episode, SimulatorError, release_for_cleanup, native_floor_adjustment
from .teacher import Teacher


def collect(client, tasks, dataset, hz=5, images=True, policy=None, teacher_probability=1.0, seed=0, progress=print, perturbation_degrees=0., absolute_targets=False,mixing_schedule=None):
    dataset = Path(dataset)
    dataset.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    coordinated_grip=bool(policy is not None and policy.metadata.get("architecture",{}).get("ramp_gripper",False))
    report_file = dataset / "collection-report.json"
    reports = json.loads(report_file.read_text())["episodes"] if report_file.exists() else []
    initial_count = len(reports)
    if not np.isfinite(perturbation_degrees) or not 0 <= perturbation_degrees <= 1:
        raise ValueError("Collection perturbation must be between zero and one degree")
    if not 0 <= teacher_probability <= 1:
        raise ValueError("Teacher mixing probability must be between zero and one")
    mixing_schedule=dict(mixing_schedule or {})
    if any(not np.isfinite(v) or not 0<=v<=1 for v in mixing_schedule.values()):raise ValueError('Stage mixing probabilities must be between zero and one')
    for number, requested in enumerate(tasks):
        writer = None
        teacher = None
        result = {"success": False, "requested_task": requested, "admissible": False}
        latencies = []
        try:
            # The teacher needs synchronized ground truth to label images; it is
            # removed before a vision policy sees an observation.
            task = configure_episode(client, {**requested,"input_mode":"state"})
            teacher = Teacher(client, task, hz)
            result["ik_prevalidated"] = True
            first = client.observe(images=images)
            if policy is not None:
                first=actor_observation(first,policy)
                policy.reset(first)
                # Warmup before acquiring the two-second lease, then reset its history.
                policy.act(first); policy.reset(first)
            provenance = "teacher_assisted" if policy is not None else "conventional"
            writer = EpisodeWriter(dataset, task, provenance, hz, {"images_each_decision": images,
                                  "teacher_source_sha256":sha256_file(Path(__file__).with_name("teacher.py")),"collection_source_sha256":sha256_file(__file__),"execution_mode":"absolute_targets" if absolute_targets else "bounded_deltas","teacher_privileged_state": True, "vision_actor_cube_pose": False, "teacher_probability": teacher_probability, "teacher_mixing_schedule": mixing_schedule, "teacher_path_recorded": True, "bounded_gripper": coordinated_grip, "perturbation_degrees": perturbation_degrees, "teacher_target_recorded": True, "teacher_version": teacher.version, "seed": seed, "simulator": getattr(client, "provenance", {})})
            client.call("rebot_experiment_control", action="start")
            client.connect(provenance=provenance, model_id="fly-brain-dagger-v1" if policy else "fly-brain-teacher-v1")
            deadline = time.monotonic() + task["timeout_seconds"] + 15
            while time.monotonic() < deadline:
                started = time.monotonic()
                observation = client.observe(images=images)
                evaluation = client.call("rebot_get_evaluation")
                if observation["phase"] not in ("running", "completed"):
                    raise RuntimeError(f"Episode ended: {observation['phase']}")
                teacher_action, label = teacher.act(observation, evaluation)
                action = getattr(teacher,"last_execution_target",teacher.last_target) if absolute_targets else teacher_action
                selected_policy = False
                intervention = getattr(teacher,"last_intervention",None)
                if policy is not None:
                    learned = policy.act(actor_observation(observation,policy))
                    probability=mixing_schedule.get(teacher.stage,teacher_probability)
                    if rng.random() >= probability:
                        action = learned
                        selected_policy = True
                if (selected_policy and any(observation["finger_contacts"].values())
                        and teacher_action["gripper_mm"] == 0 and action["gripper_mm"] > 0):
                    intervention = {"proposed_policy_action": action, "reason": "teacher-assisted prevention of premature release"}
                    action = {**action, "gripper_mm": 0}
                    result["gripper_interventions"] = result.get("gripper_interventions", 0) + 1
                if perturbation_degrees and not selected_policy and teacher.stage in ('approach','pregrasp') and rng.random()<.35:
                    from .adapters import JOINT_LOWER,JOINT_UPPER
                    sigma=perturbation_degrees
                    if teacher.stage in ('descend','close'):sigma*=.3
                    if any(observation['finger_contacts'].values()):sigma*=.03
                    perturbation=rng.normal(0,sigma,6)
                    intervention={**(intervention or {}),'teacher_action_before_perturbation':action,'collection_perturbation_deg':perturbation.tolist()}
                    action={**action,'joints_deg':np.clip(np.asarray(action['joints_deg'])+perturbation,JOINT_LOWER,JOINT_UPPER).tolist()}
                if coordinated_grip:
                    action={**action,"gripper_mm":observation["gripper_mm"]+float(np.clip(action["gripper_mm"]-observation["gripper_mm"],-25/hz,25/hz))}
                try:
                    client.action(observation, action)
                except SimulatorError as error:
                    # A definitive floor rejection consumed no action ID. In this
                    # explicitly teacher-assisted run, execute the correction and
                    # retain the rejected proposal in the dataset. Never do this
                    # for timeouts, uncertain responses, or autonomous evaluation.
                    if not selected_policy or "endpoint intersects the floor" not in str(error):
                        raise
                    intervention = {"rejected_policy_action": action, "reason": str(error)}
                    action = teacher_action
                    client.action(observation, action)
                    result["floor_interventions"] = result.get("floor_interventions", 0) + 1
                native_projection = native_floor_adjustment(client, action)
                if native_projection:
                    intervention = {**(intervention or {}), **native_projection}
                    result["floor_limited_actions"] = result.get("floor_limited_actions", 0) + 1
                if policy is not None:
                    intervention={**(intervention or {}),'policy_selected':selected_policy,'teacher_probability':probability}
                latency_ms = (time.monotonic() - started) * 1000
                latencies.append(latency_ms)
                writer.append(observation, action, label, teacher.stage, evaluation, latency_ms, intervention,
                              expert_target=getattr(teacher, "last_target", None),expert_path=teacher.path)
                if teacher.stage == "done":
                    result.update(success=True, hold_success=teacher.hold_succeeded, release_success=True,
                                  admissible=True, final_evaluation=evaluation)
                    break
                minimum_settle=1/hz+.05 if policy is not None and policy.metadata.get("architecture",{}).get("settle_before_observation",False) else 0.
                time.sleep(max(minimum_settle, 1 / hz - (time.monotonic() - started)))
            else:
                raise TimeoutError("Collection exceeded its bounded episode time")
        except (OSError, RuntimeError, ValueError, TimeoutError) as error:
            result["error"] = str(error)
            if writer is not None:
                # A kinematically plausible task is not admitted to the solved workspace until contact succeeds.
                result["teacher_failure"] = True
        finally:
            if teacher is not None:
                result["hold_success"]=teacher.hold_succeeded
                result["release_success"]=teacher.stage=="done" and teacher.hold_succeeded
            cleanup_error = release_for_cleanup(client)
            if cleanup_error:
                result["cleanup_error"] = cleanup_error
            if latencies:
                result["p95_loop_ms"] = float(np.percentile(latencies, 95))
                result["max_loop_ms"] = max(latencies)
                result["cadence_passed"] = result["p95_loop_ms"] < 1000 / hz
            if writer is not None:
                result["episode_directory"] = str(writer.finish(result))
            reports.append(result)
            write_json(dataset / "collection-report.json", {"episodes": reports, "successes": sum(r["success"] for r in reports)})
            progress(f"Episode {initial_count + number + 1}: {'success' if result['success'] else result.get('error', 'failed')}", flush=True)
    return {"episodes": len(reports), "new_attempts": len(reports) - initial_count, "successes": sum(r["success"] for r in reports), "dataset": str(dataset)}


def actor_observation(observation,policy):
    """A vision student never receives the teacher's cube-pose supervision."""
    if policy.mode!='vision':return observation
    return {**{k:v for k,v in observation.items() if k!='cube_pose'},'input_mode':'vision'}
