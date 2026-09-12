> Historical experiment note. For the current published two-camera controller and model downloads, start with [README.md](README.md) and [training.md](docs/training.md). Referenced local reports/data are not all included in the public release.

# Camera-driven MaleCNS response experiment

This implements the first response curriculum: see a 20 mm cube and turn the
robot base one degree toward it. It does **not** pick up, lift, or hold the cube.
The previous simulator-state controller remains available separately.

## Run in the UI

Open `/Users/monomyth/github/rebot-motion-lab-codex/dist/ReBot Motion Lab Codex.app`.

1. Use **Load task…** and select
   `/Users/monomyth/code/codex/fly-brain/configs/retinal-cube20-response.json`.
   This places a 20 mm cube at X=350, Y=12 mm and starts the arm folded, gripper closed.
2. Use **Choose model…** and select:

   `/Users/monomyth/code/codex/fly-brain/data/checkpoints/rebot/retinal-response-final-20260910-140300/seed-0`

3. Leave **Learn from reward** off for a deterministic test. Click **Run fly brain**.
   Expect one small base turn. The brain overlay displays computed neuron activity
   and the PAM01 reward signal while the controller owns the episode.
4. Reset before repeating. To test the opposite direction, change cube Y to -12,
   apply the cube, then reset. Keep X around 338–363 mm and Y within ±8–15 mm.
5. Enable **Learn from reward** to explore actions and update eligible synapses.
   Each completed training invocation saves a new checkpoint under the run's
   shared Results folder; the original weights remain intact. The UI selects
   the new checkpoint. A single trial can make performance worse. Reselect the
   evaluated model above to return to fixed-model testing.

The controller is qualified only for the small response experiment from its
folded decision pose. Arbitrary cube placement, size, and arm poses are not yet
supported by a validated image-based policy. A cube at Y=0 is already aligned
and does not exercise this task. A one-degree response is intentionally small.

## Terminal

```sh
cd /Users/monomyth/code/codex/fly-brain
export MALECNS_HOME=/Users/monomyth/code/codex/fly-brain/data
.venv/bin/fly-brain visual-run \
  --checkpoint "/Users/monomyth/code/codex/fly-brain/data/checkpoints/rebot/retinal-response-final-20260910-140300/seed-0" \
  --episodes 2 \
  --output "$MALECNS_HOME/runs/retinal-response-$(date +%Y%m%d-%H%M%S)"
```

The two default episodes use Y=+12 and -12 mm, cube20, folded pose. Terminal runs
open and close their own simulator instances. UI runs use the current window.
Add `--learn` to enable exploration and PAM-gated plasticity, saving a new
checkpoint at `OUTPUT/checkpoint`. Without it, a reward pulse is displayed but
synaptic weights remain fixed. Neither path calls the scripted pickup teacher.

## What drives the action

```text
Front + Top RGB cameras
  → luminance samples at 3,316 annotated R1–R6 photoreceptors
  → signed MaleCNS sensory connections
  → 206 visual KCg-d Kenyon cells
  → 892 plastic recorded KCg-d→MBON edges / 11 MBONs
  → 43 descending neurons
  → 103 annotated leg motor neurons
  → fixed tibial extensor-minus-flexor population readout
  → one-degree robot base turn
```

The actor receives camera pixels to select direction. Joint angles and aperture
are used to apply a relative action to the current robot pose. It does not receive
cube coordinates, detections, bounding boxes, desired joint angles, or a teacher
label. Ground-truth geometry exists only in the task setup and reward evaluator,
which scores whether the actual turn reduced angular error. The experiment harness
also uses that score to stop a trial; a visual-only stopping policy is not implemented. Training necessarily
uses that task-dependent reward signal.

Photoreceptors are registered to visual columns using their strongest outgoing
L1/L2 column vote. Left-eye columns are assigned to Front, right-eye columns to
Top. These are explicitly engineered camera registrations, not reconstructed fly
viewing directions. Uniform luminance gain and train-only mean adaptation are the
validated defaults. `--local-contrast` is an experimental alternative tested with
moving-arm images; it is not promoted to the response baseline.

Fast synaptic signs use transmitter predictions: ACh positive, histamine/GABA
negative, and central glutamate assumed negative. Neuromodulators have no direct
fast drive in the sensory matrix. Unknown fast transmitters default positive.
These signs and the abstract tanh rate-deviation dynamics are modelling choices;
the connectome does not supply all receptors, membrane dynamics, or learned weights.
Sensory states reset per camera presentation, and output feedback is excluded.
The output path currently uses only direct MBON→DN→motor connections.

## Dopamine and learning

Forty-four annotated PAM01 dopamine cells provide a signed deviation from tonic
activity, `tanh(reward - expected_reward)`. Positive reward increases their signal;
unexpected failure can produce a dip. Their recorded connections to the selected
MBONs gate updates of eligible KCg-d→MBON synapses.

Eligibility is presynaptic activity multiplied by exploratory postsynaptic
perturbation, decayed over the measured reward delay. Only the 892 existing edges
can change, and weights stay between zero and four times their initial normalized
value. The exponential reward expectation updates at 0.02 per response. There is
no PPO/Adam optimizer in this new controller and no trainable motor readout.

This is an engineered three-factor learning rule, **not** a validated simulation
of gamma5 molecular plasticity. It proves that reward-modulated changes affect
this model's behaviour; it does not prove that the real fly learns this way.

## Rebuild the response dataset and train

No additional model download is needed. Existing shared MaleCNS files and soma
geometry are reused. Keep the Mac unlocked and awake for native physics.

```sh
cd /Users/monomyth/code/codex/fly-brain
export MALECNS_HOME=/Users/monomyth/code/codex/fly-brain/data
STAMP=$(date +%Y%m%d-%H%M%S)
DATASET="$MALECNS_HOME/datasets/visual-dopamine/response-$STAMP"
MODELS="$MALECNS_HOME/checkpoints/rebot/retinal-response-$STAMP"

.venv/bin/fly-brain visual-prepare
.venv/bin/fly-brain visual-collect --output "$DATASET"
.venv/bin/fly-brain visual-train \
  --dataset "$DATASET" --output "$MODELS" --trials 8000 --seeds 0 1 2
.venv/bin/fly-brain visual-run --checkpoint "$MODELS/seed-0" --episodes 2 \
  --output "$MALECNS_HOME/runs/retinal-evaluation-$STAMP"
```

Collection records 24 camera scenes and physically tries both base directions,
resetting to folded before each trial. This conventional exploration produces a
bank of measured action rewards; it is not demonstration imitation. Sixteen
scenes train and eight held-out scenes test. Neural training replays camera-driven
features against the measured reward for its sampled action. The 8,000 trials
are reward-bank samples, not 8,000 newly simulated physical trials. Each of three
seeds also runs an otherwise matching dopamine-disabled control.

`--base-prefixes -2 -1 0 1 2` collects the harder changing-view curriculum.
Every episode still starts folded, then an explicitly recorded conventional
base-turn prefix establishes the decision pose. `visual-run --max-steps 4` tests
repeated neural decisions; this extension is not yet reliable.

## Inspect data and learned changes

- Dataset `case-*/Front.jpg` and `Top.jpg`: actual camera inputs, checked by SHA-256.
- `observation.json`: images and proprioception/control timing, excluding cube pose.
- `outcomes.json`: separately identified evaluator geometry, tried actions and rewards.
- `collection.json`: scene splits and simulator provenance.
- Model `training.json`: learning curve on training cases.
- `dopamine-updates.jsonl`: each sampled response, reward, dopamine and weight change.
- `comparison.json` in the model's parent folder: before/after, three seeds,
  dopamine-disabled and visual-input-blocked controls.
- Run `evaluation.json`, `response-*/step-*/decision.json`, and JPEGs: fresh
  physical outcomes, action, measured joint pose and reward.

`model.npz` is a non-pickle NumPy archive containing the plastic weights and
visual calibration. `manifest.json` records circuit/graph IDs, assumptions,
learning settings and file hashes. Neuron body IDs, types, and population membership
are in shared `circuits/CIRCUIT_ID/neurons.json` and `populations.npz`.

## Share with another project

Set `MALECNS_HOME=/Users/monomyth/code/codex/fly-brain/data` and install this Python package
in that project's environment. Load
`fly_brain.visual_dopamine.policy.VisualDopaminePolicy(checkpoint, root)`.
All large assets stay in shared storage. This robot's camera/motor registration
must be adapted for another body; the checkpoints are not universal robot skills.

`fly-brain export CHECKPOINT DESTINATION` exports the checkpoint while referencing
the same shared assets. Add `--include-graph` for a self-contained data bundle,
then set `MALECNS_HOME` to the bundle. It includes attribution and the derived
circuit. It does not bundle the Python environment or native simulator.

## Evidence and next boundary

See `reports/visual-dopamine/RESULTS.md` for measured results and failures. The next
research step is robust multi-step visual alignment, then additional joint command
pairs, reaching/contact rewards, grasp/lift and delayed hold credit. None of those
later tasks is claimed complete by this base-response checkpoint.

Biological references: [visual Kenyon cells](https://elifesciences.org/articles/14009),
[mushroom-body wiring](https://elifesciences.org/articles/62576),
[leg motor units](https://elifesciences.org/articles/56754), and
[MaleCNS data](https://male-cns.janelia.org/download/).
