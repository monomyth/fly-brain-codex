# Training and reproduction

The published deployed controller uses Front/Gripper images and robot feedback. Older `train-bc --mode state` examples describe a different experiment.

## Reproduce the latest head fit

The HF release includes `training-bundle.tar.gz`, a compact frozen-feature snapshot with initial neural parameters, fixed wiring, exact targets, train/validation/test membership, parameter scaling and the matching trainer. It avoids moving the full graph and RGB corpus to the training GPU. It is sufficient for the published continuation objective; it cannot create new visual examples.

Download it at the commit in `configs/model-release.json`, verify the archive against `release.json`, and unpack it into a new project directory such as `~/code/codex/fly-brain/data/exports/published-training`. On a Linux machine with a compatible NVIDIA GPU and CUDA PyTorch environment:

```sh
cd ~/code/codex/fly-brain/data/exports/published-training
python benchmark_training_bundle.py
python continue_head_training.py --iterations 12000 --output ./fit-new
```

The trainer requires a passing CPU/CUDA loss-and-gradient comparison, verifies bundle hashes, preserves connection signs/masks and parameter bounds, and selects parameters on validation rows. Test rows do not select parameters. The recorded fit used PyTorch 2.14.0+cu130 and an RTX 4090. Exact optimizer results can differ across hardware/library versions.

The archive on HF called `cuda-continuation.tar.gz` is the recorded output, not a qualified simulator model. The newer touch-gated model is a separate artifact with separate native evaluation.

## Collect new two-camera observations

Build the simulator and download the runtime assets first. Keep macOS unlocked and its display awake. The following collector starts folded with a 20 mm cube, stores paired Front/Gripper RGB, measured body feedback and teacher actions, and declares whole-position train/validation/test splits:

```sh
.venv/bin/python scripts/collect_two_view_pickups.py --run reports/my-two-view --limit 1
# Inspect the first case before collecting the rest:
.venv/bin/python scripts/collect_two_view_pickups.py --run reports/my-two-view
```

Data go under `data/datasets/rebot-pick/my-two-view`. Failed attempts stay available for diagnosis. This is teacher collection, not neural policy execution. Do not use `--no-images` for this pipeline.

New datasets require corresponding training configurations and visual normalization. Inspect `.venv/bin/fly-brain feedback-train --help` for the supported camera/body training options, including temporal sensory state and constrained motor-synapse fitting. Root-level research scripts describe recovery collection and previous loss experiments; many require their referenced private historical datasets and reports.

## Before deployment

Import a new fit into a new checkpoint with `scripts/import_continued_head.py`; check saved-head numerical agreement with `scripts/check_imported_head.py`. Full sensory parity with `scripts/benchmark_mlx_sensory.py` requires the original images named in that checkpoint, which are not included in this public release. A fresh dataset supplies a new complete parity reference.

Then evaluate a frozen policy in fresh native trials. The UI gate requires a completed 20-trial qualification with at least 18 qualifying holds, matching model/runtime hashes and timing. The full task should additionally verify the scripted release and floor landing. A better training loss or successful teacher trajectory does not satisfy that gate.

All newly derived observations, caches, checkpoints and remote copies stay under a directory containing `codex`. Shared storage is reserved for original reference downloads. The release's historical qualification records describe the original Mac; install-time relocation is not a new qualification on another machine.
