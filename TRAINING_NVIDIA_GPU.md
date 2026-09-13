> Historical experiment note. For the current published two-camera controller and model downloads, start with [README.md](README.md) and [training.md](docs/training.md). Referenced local reports/data are not all included in the public release.

# Training on NVIDIA GPU host

NVIDIA GPU host trains the learned motor-control portion of the MaleCNS-derived network using PyTorch/CUDA on its RTX 4090. The Mac supplies rendered Front/Gripper observations and performs RealityKit evaluation. The GPU does not run the macOS simulator.

All remote code, caches, training inputs and outputs live under `$HOME/code/codex/fly-brain`. All local generated artifacts live under `$HOME/code/codex/fly-brain`. Original downloaded connectome data remain in shared storage; generated data do not go there.

## Current verified experiment

`reports/cuda-continuation-20260911/` contains the benchmark, training trace, import checks and simulator result. Its `STATUS.json` reports progress and qualification. The candidate checkpoint is `data/checkpoints/rebot/cuda-continuation-20260911`; it is not automatically selected in the UI.

The portable input bundle is `data/exports/gpu-server-priority-20260911-151700`, mirrored remotely to `experiments/priority-recovery-20260911-151700`. Its approximately 7.8 MB contain frozen sensory features, exact teacher targets, training-only weights, split identifiers, initial learned parameters and fixed neural-head wiring. This compact bundle is sufficient for continued head fitting; collecting new observations still requires the Mac and its original images.

CUDA and CPU loss/gradient agreement was verified before training. Forward/backward of the full 1,904-row training split took about 10.8 ms on the 4090 versus 602 ms on the same Linux machine's CPU. This is a step benchmark, not an end-to-end training speedup.

## Repeat a fit from that bundle

Run from the local project. Choose a new output name for every fit; existing checkpoints and experiment outputs are never overwritten.

```sh
cd "$HOME/code/codex/fly-brain"
# Configure gpu-server as your SSH alias; the remote project is home-relative.
rsync -a scripts/continue_head_training.py \
  gpu-server:code/codex/fly-brain/experiments/priority-recovery-20260911-151700/
ssh gpu-server 'cd "$HOME/code/codex/fly-brain" && \
  .venv-cuda/bin/python experiments/priority-recovery-20260911-151700/continue_head_training.py \
  --iterations 8000 \
  --output "$HOME/code/codex/fly-brain/experiments/priority-recovery-20260911-151700/continued-new-run"'
```

The trainer verifies bundle hashes, retains existing anatomical connections and signs, enforces the original parameter bounds, and selects weights with the equal legacy/recovery validation score. Test rows never choose weights. All 8,000 iterations run, but the selected weights can be from an earlier validation checkpoint.

Copy the new output back into a new project report directory, then use `scripts/import_continued_head.py --source ... --result ... --output ...`. Run `scripts/check_imported_head.py` and `scripts/benchmark_mlx_sensory.py` for that exact checkpoint before `scripts/run_mlx_checkpoint.py`. See `reports/cuda-continuation-20260911/evaluate-command.json` for the recorded simulator invocation.

Neither an improved fitting loss nor numerical agreement proves successful pickup. Qualification requires fresh native simulation trials with the unchanged 20 mm cube, folded start, 100 mm clearance, 5-degree tilt tolerance, and 5-second hold. The actor receives camera images and proprioception, with no teacher, cube coordinates or task phase. Privileged geometry supplies training targets and evaluator measurements only.

## Configuring your GPU machine

The recovery-training script accepts `--host` and `--remote-project`; `FLY_BRAIN_GPU_HOST` and `FLY_BRAIN_GPU_PROJECT` also work. You can instead store `{"host":"gpu-server","project":"code/codex/fly-brain"}` in ignored `data/gpu-host.json`. Public scripts have no personal SSH hostname or absolute user-home default.
