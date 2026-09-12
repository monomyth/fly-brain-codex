# CUDA training on blacktower

The isolated Linux project is `/home/monomyth/code/codex/fly-brain`.
It uses `.venv-cuda` with PyTorch 2.14.0 and CUDA 13.0 on the RTX 4090.
MLX is not used. The simulator remains on the Mac.

The installed NVIDIA kernel module and NVML library have different versions, so
`nvidia-smi` currently reports a mismatch. Actual CUDA initialization, allocation,
forward propagation and backpropagation passed. No drivers were changed.

The measured benchmark is in `reports/gpu-benchmark/blacktower-results.json`:

- Fixed sensory graph, 16-frame batch: 0.354 seconds/frame with SciPy on the
  Linux CPU versus 0.00180 seconds/frame with CUDA.
- Motor forward/backward, 128-frame batch: 0.0194 seconds/iteration on that CPU
  versus 0.00159 seconds/iteration with CUDA.
- Maximum CPU/GPU feature difference: 0.000000626.

These are operation timings, not a claim that the entire training job is 197x
faster. Optimization still uses SciPy L-BFGS-B, with PyTorch gradients on CUDA.

Directories on blacktower:

- `src/`: project Python source.
- `data/benchmark-bundle/`: verified prepared graph/circuits and a reference
  checkpoint used for feature calibration and benchmark reproduction.
- `data/training-datasets/`: explicitly authorized copies of the generated camera
  demonstrations. The manifests identify conventional and teacher-assisted runs.
- `data/checkpoints/rebot/`: CUDA-trained checkpoints.
- `reports/`: training logs, dataset splits, and benchmark results.
- `.cache/`: project-specific runtime caches for subsequent commands.

All new research outputs on the Mac belong in this project's `data/` or
`reports/`, not in `/Users/monomyth/code/data/malecns`. That shared directory is
reserved for original downloads. CUDA checkpoints use the same NumPy format as
local checkpoints and can be copied back and loaded on the Mac.

The `feedback-train` command exposes the new trainer. A Linux invocation uses
`PYTHONPATH=src .venv-cuda/bin/python -m fly_brain.cli --home data/benchmark-bundle
feedback-train ... --device cuda --learn-motor-synapses`. The exact executed
training wrappers and split specifications are retained in `scripts/` and
`reports/pickup-fix/` on the Mac and in the Linux project.

The optional SSH sensory worker sends lossless control features and the same
8-bit neural activity values already used by the brain overlay. It opens no
listening network service. Its feature equivalence and measured round-trip
latency are recorded in `reports/pickup-fix/remote-sensory-validation.json`.

Model accuracy on recorded demonstrations does not establish autonomous pickup
success. Refer to the corresponding native `evaluation.json` before promoting a
checkpoint to the simulator UI.

## Local fallback for the approved two-view experiment

For the 2026-09-11 continuation, the user confirmed blacktower is down. The
Front/Gripper experiment therefore runs locally with PyTorch CPU training.
Its data, exact commands, numerical-equivalence benchmark, and results are in
`reports/two-view-20260910-233652/`. The CPU feature cache can support further
local fits without repeatedly propagating the same sensory frames. No MLX
backend or host driver changes were introduced.
