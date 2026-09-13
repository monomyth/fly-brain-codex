# Training with an NVIDIA GPU

The learned motor head can be fitted on a Linux machine with an NVIDIA GPU and a compatible PyTorch/CUDA installation. The native RealityKit simulator and Front/Gripper observation collection run on the Mac. MLX is used for the Mac's sensory inference, while PyTorch/CUDA performs GPU fitting.

The recorded GPU experiments used an RTX 4090, PyTorch 2.14.0 and CUDA 13.0. These are the measured reference environment, not a requirement to use a particular machine name. See [training and reproduction](docs/training.md) for the downloadable frozen-feature bundle and verification steps.

## Configure your machine

Use an SSH alias such as `gpu-server`, and keep the remote project under `~/code/codex/fly-brain`. The recovery-training wrapper accepts:

```sh
.venv/bin/python scripts/train_runtime_recovery.py \
  --plan reports/my-experiment/plan.json \
  --host gpu-server \
  --remote-project code/codex/fly-brain
```

`FLY_BRAIN_GPU_HOST` and `FLY_BRAIN_GPU_PROJECT` are alternatives to the flags. You can also use ignored `data/gpu-host.json`:

```json
{"host":"gpu-server","project":"code/codex/fly-brain"}
```

The project path is resolved on the remote machine. Generated observations, caches, checkpoints and logs stay inside each project's `data/` or `reports/` directories. The shared reference-data directory contains original downloads only.

The optional stateless CUDA sensory worker also accepts an explicit SSH host. Persistent temporal inference currently uses the local MLX backend; fitting on a GPU does not change that deployment contract.

CPU/GPU arithmetic checks and fresh native trials are required before promoting a fitted model. Better fitting loss alone does not establish successful pickup.
