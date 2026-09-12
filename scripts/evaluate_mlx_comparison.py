"""Recorded MLX comparison of the unchanged temporal checkpoint and task battery."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from fly_brain.assets import sha256_file, write_json
from fly_brain.simulator import EpisodeSimulator
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.motor_run import run
from mlx_sensory_backend import MLXFeedbackCore


def main():
    root = Path(__file__).resolve().parents[1]
    batch = root / "reports/temporal-control-20260911"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=batch / "full-temporal-mlx-20")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("Use a new output directory within the fly-brain project")
    checkpoint = root / "data/checkpoints/rebot/comparison-temporal-v1-20260911"
    parity_path = batch / "mlx-parity-full.json"
    parity = json.loads(parity_path.read_text())
    if (not parity["passed"] or Path(parity["checkpoint"]) != checkpoint
            or parity["frames"] != 1975 or parity["episodes"] != 42):
        raise ValueError("The complete offline equivalence check must pass first")
    tasks_path = batch / "full-tasks-20.json"
    tasks = json.loads(tasks_path.read_text())
    if len(tasks) != 20:
        raise ValueError("Expected the unchanged 20-task battery")
    output.mkdir(parents=True)
    model_hash = sha256_file(checkpoint / "model.npz")
    plan = {
        "checkpoint": str(checkpoint), "model_sha256": model_hash,
        "tasks_sha256": sha256_file(tasks_path), "parity_report": str(parity_path),
        "parity_sha256": sha256_file(parity_path),
        "backend_sha256": sha256_file(root / "scripts/mlx_sensory_backend.py"),
        "backend": "mlx-metal-0.32.2", "learn": False, "max_steps": 160,
        "settle_seconds": 0.1, "control_hz": 2,
        "promotion_minimum_successes": 18,
        "scope": "Runtime-delay comparison; unchanged weights, task list, sensory equations, camera geometry and native success criteria",
    }
    write_json(output / "plan.json", plan)
    guard = subprocess.Popen(
        ["/usr/bin/caffeinate", "-di", "-w", str(os.getpid())],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    write_json(output / "process.json", {"pid": os.getpid(), "power_guard_pid": guard.pid})
    try:
        policy = MotorPolicy(checkpoint, root / "data")
        if policy.hz != 2:
            raise ValueError("Unexpected control frequency")
        policy.core = MLXFeedbackCore(policy.core)
        with EpisodeSimulator(log=output / "native.log") as client:
            report = run(client, policy, tasks, output, max_steps=160, learn=False, settle_seconds=0.1)
        unchanged = sha256_file(checkpoint / "model.npz") == model_hash
        result = {
            "status": "completed" if report["attempts"] == 20 else "interrupted",
            "backend": plan["backend"], "attempts": report["attempts"],
            "successes": report["successes"], "weights_unchanged": unchanged,
            "promotion_target_met": unchanged and report["attempts"] == 20 and report["successes"] >= 18,
        }
        write_json(output / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if guard.poll() is None:
            guard.terminate()
            guard.wait(timeout=5)


if __name__ == "__main__":
    main()
