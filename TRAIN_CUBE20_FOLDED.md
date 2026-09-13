> Historical experiment note. For the current published two-camera controller and model downloads, start with [README.md](README.md) and [training.md](docs/training.md). Referenced local reports/data are not all included in the public release.

# Train 20 mm cube pickup from the folded idle pose

**Input mode:** this recipe trains on simulator state, including the cube’s measured position relative to the gripper. It does not train perception from images. `--no-images` also disables saving camera frames. Camera-based training needs a new image-recorded dataset and `train-bc --mode vision`; the existing vision adapter uses orange-cube image features plus robot proprioception.

**Ready for collection:** simulator 1.8 fixes the fingertip collision contacts and gripper surface calibration. Ten of ten teacher demonstrations passed from fully folded starts, including all four corners and random positions in the collection area. Every trial picked up a 20 mm cube, lifted it at least 100 mm, held it level for five seconds, and released it. The dataset audit and training loader passed. See `reports/teacher-fix-verification/result.json`.

Every demonstration starts with all six joints at 0 degrees and the gripper closed (0 mm). The simulator settles in that pose, then the teacher starts controlling the arm. The dataset includes unfolding, approach, grasp, lift, a five-second hold, and release. This differs from the collector's default Ready pose, so always pass the explicit task file below.

The reusable native task is `configs/cube20-folded.json`. It is also suitable for the simulator's Load task button. The collection commands vary the cube position around (350, 0) mm while keeping the size at 20 mm, yaw at zero, and the folded start constant.

Run the following blocks in the same terminal. Keep macOS unlocked and the display awake during collection and evaluation. They launch isolated Codex simulator instances. No extra downloads are required. Full model training can take hours. The ten verification demonstrations are saved in shared storage. No 100-episode collection or model training has been started by Codex.

## Optional single-trial check

```bash
cd $HOME/code/codex/fly-brain
export MALECNS_HOME=$HOME/code/codex/fly-brain/data
.venv/bin/python - <<'PYTASK'
import json
from pathlib import Path
base = json.loads(Path("configs/cube20-folded.json").read_text())
Path("configs/cube20-folded-check.json").write_text(json.dumps([base], indent=2) + "\n")
PYTASK

.venv/bin/fly-brain collect \
  --dataset "$MALECNS_HOME/datasets/rebot-pick/cube20-folded-check-$(date +%Y%m%d-%H%M%S)" \
  --tasks configs/cube20-folded-check.json \
  --hz 5 --no-images
```

This verifies one conventional-teacher demonstration from folded idle. Expect `successes: 1` with the updated app. The next blocks collect a fresh dataset and train a new checkpoint.

## 1. Create a fresh shared dataset and explicit collection tasks

```bash
cd $HOME/code/codex/fly-brain
export MALECNS_HOME=$HOME/code/codex/fly-brain/data
FLY_RUN="cube20-folded-$(date +%Y%m%d-%H%M%S)"
FLY_DATA="$MALECNS_HOME/datasets/rebot-pick/$FLY_RUN"
FLY_MODEL="$MALECNS_HOME/checkpoints/rebot/$FLY_RUN"
FLY_REPORT="$MALECNS_HOME/runs/$FLY_RUN"
mkdir -p "$FLY_REPORT"

.venv/bin/python - "$FLY_REPORT/collection-tasks.json" <<'PYTASKS'
import json, random, sys
from pathlib import Path
base = json.loads(Path("configs/cube20-folded.json").read_text())
rng = random.Random(2000)
tasks = [
    {**base,
     "cube_xy_mm": [350 + rng.uniform(-15, 15), rng.uniform(-15, 15)],
     "seed": 2000 + i}
    for i in range(100)
]
assert all(t["initial_joints_deg"] == [0] * 6 and
           t["initial_gripper_mm"] == 0 and t["cube_size_mm"] == 20
           for t in tasks)
Path(sys.argv[1]).write_text(json.dumps(tasks, indent=2) + "\n")
print("Created 100 tasks: 20 mm cube, folded joints, closed gripper.")
PYTASKS
```

Positions cover X=335–365 mm and Y=-15–15 mm. This is a small training area; it does not establish generalization to arbitrary floor positions. The task file controls the episode count, size and initial pose, so `--episodes` and `--sizes` are unnecessary when collecting it. Do not omit `--tasks`: the normal default starts at Ready.

## 2. Collect and check demonstrations

```bash
.venv/bin/fly-brain collect   --dataset "$FLY_DATA"   --tasks "$FLY_REPORT/collection-tasks.json"   --hz 5 --no-images

.venv/bin/fly-brain audit-dataset "$FLY_DATA"   --output "$FLY_REPORT/audit.json"
```

Check that the teacher is completing pickups and the audit reports `"valid": true`. Repeated teacher failures should be investigated before training. Failed demonstrations are retained for diagnosis and excluded from the default training split. The state-based policy does not need camera images. The brain overlay remains idle during this conventional-teacher collection because no MaleCNS policy is driving the robot.

```bash
.venv/bin/fly-brain split "$FLY_DATA" --seed 0
.venv/bin/fly-brain tasks "$FLY_DATA"   --split test --output "$FLY_REPORT/test-tasks.json"
```

The split separates whole cube configurations and starting poses. Training requires successful examples in both train and validation. Test tasks retain the folded initial pose.

## 3. Train a fresh MaleCNS checkpoint

```bash
.venv/bin/fly-brain train-bc   --dataset "$FLY_DATA" --output "$FLY_MODEL"   --kind malecns --mode state   --channels 2 --updates 2   --epochs 10 --sequence-length 32   --learning-rate 0.0003 --seed 0
```

This fits new normalization and trains the sensory adapter, shared neuron-response parameters, and motor decoder using the fixed MaleCNS graph. Existing models stay in place. The new output directory must not already exist. The best validation-loss weights are saved, but validation loss is not autonomous task success.

## 4. Test autonomous pickup from folded

```bash
.venv/bin/fly-brain evaluate   --checkpoint "$FLY_MODEL"   --tasks "$FLY_REPORT/test-tasks.json"   --floor-projection   --output "$FLY_REPORT/evaluation"
```

Inspect `successes` and `attempts` in the report. There is no teacher control in this evaluation. A failed learned pickup means further diagnosis or correction data are needed; one successful teacher trial does not guarantee that the learned policy will succeed.

## 5. Use it in the UI

1. Stop the current run. **Load task…** is available in the cube panel even when a task is already configured. The toolbar’s **Import task or trajectory** also accepts task files.
2. Click **Load task…** and select `$HOME/code/codex/fly-brain/configs/cube20-folded.json`.
3. Wait for the scene to settle. The status says **Ready**, but the arm is physically folded, with all six joints zero and the gripper closed.
4. Click **Choose model…** and select the checkpoint directory printed by training (`$FLY_MODEL`).
5. Click **Run fly brain**.
6. After a run, or after moving/resizing the cube, click the experiment's **Reset** before the next attempt. With this task loaded, Reset returns to folded and preserves the configured cube placement. Applying the cube alone preserves the arm's current pose.

Use a 20 mm cube within the collected area initially. No app rebuild is needed to select a newly trained model.

## Recoverable floor contact

External controller targets that would cross the floor are limited at contact. The running episode and controller session stay active, so the next upward action can raise the arm. Native floor adjustments are recorded in collection/evaluation data. Normal timeout and invalid-command checks still apply.

The verified collection recipe is for 20 mm cubes in X=335–365 mm, Y=-15–15 mm, yaw=0, starting folded. A separate 50 mm folded-start compatibility trial passed. The separate 10 mm and 90 mm folded-start trials failed, so those sizes are not qualified by this fix. Cube placement/resizing still supports the existing 10–90 mm range.
