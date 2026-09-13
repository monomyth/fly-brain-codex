# MaleCNS motor-neuron mapping for the ReBot arm

This document records the original seven-channel mapping and its early response experiment. The current Front/Gripper pickup controller adds body-sensory inputs and expanded trainable pathways; its latest evidence is in [the two-view experiment](reports/two-view-20260910-233652/EXPERIMENT.md) and [status](reports/two-view-20260910-233652/STATUS.json). Candidate LTM grip and MN7 opening alternatives are recorded in [GRIPPER_NEURON_CANDIDATES.md](GRIPPER_NEURON_CANDIDATES.md).

The seven robot channels use **75 identified motor neurons in fourteen disjoint
opposing pools**. Positive/negative mean increasing/decreasing the robot joint
angle, or opening/closing its gripper. This is an engineered body mapping, not a
claim that a robot joint is anatomically equivalent to a fly joint.

| Robot channel | Fly leg | Positive muscle pool | Negative muscle pool | Neurons (+ / −) |
|---|---|---|---|---:|
| J1 | Left front | Sternal anterior rotator MN | Sternal posterior rotator MN | 2 / 4 |
| J2 | Left front | Tr extensor MN | Tr flexor MN, Acc. tr flexor MN | 2 / 11 |
| J3 | Left front | Ti extensor MN | Ti flexor MN, Acc. ti flexor MN | 2 / 15 |
| J4 | Right front | Sternal anterior rotator MN | Sternal posterior rotator MN | 2 / 2 |
| J5 | Right front | Tr extensor MN | Tr flexor MN, Acc. tr flexor MN | 2 / 10 |
| J6 | Right front | Ti extensor MN | Ti flexor MN, Acc. ti flexor MN | 2 / 14 |
| gripper | Left front | Ta levator MN | Ta depressor MN | 2 / 5 |

Tr = trochanter; Ti = tibia; Ta = tarsus. The full body-ID list and individual
readout coefficients are in [neuron-map.csv](reports/motor-control/neuron-map.csv).
The immutable shared mapping is `$HOME/code/codex/fly-brain/data/motor-circuits/4ba94c49dd7f3e6e6243101698dfc037d45e7d681377d587b5c1bc0b4cc9ebf7/mapping.json`.

## Signal route

Camera pixels → annotated photoreceptors → visual KCg-d cells → plastic KC→MBON
synapses → MBONs → central-brain relays → descending neurons → VNC premotor
neurons → mapped motor neurons → robot channels.

Direct MBON→DN and DN→MN contacts are also retained. The expanded output path has
2,452 central-brain relays, 925 descending neurons, and 1,994 VNC premotor neurons.
It contains only recorded connections. Incoming contact counts are normalized
within each layer and transmitter-sign assumptions are inherited from the retinal
experiment. The seven-channel small-signal transfer has rank seven, so the route
can span seven independent directions around its operating point. This does not
establish independent behavioural control for arbitrary large neural states.

Each command is the mean activity of its positive pool minus the mean activity
of its negative pool. Gains come from neural perturbation sensitivity, without
cube labels. The bridge weights and gains remain fixed during training. The
892 existing visual KCg-d→MBON edges remain the only plastic synapses, gated by
44 PAM01 dopamine neurons. The final learner uses motor-pool exploration and
chain-rule synaptic eligibility through the fixed output circuit. This credit
calculation, tonic KC drive, and motor rest reference are engineering choices,
not established fly learning physiology. Shared circuit ID:
`4ba94c49dd7f3e6e6243101698dfc037d45e7d681377d587b5c1bc0b4cc9ebf7`.

## Physical mapping check

**14/14 direct-stimulation trials passed.** Each pool moved only its assigned
robot channel: ±1 degree for a joint or ±2 mm for the gripper. Each calibration
scene began folded; a recorded setup pose `[0,-10,-10,0,0,0]`, gripper40, allowed
both directions at joints that begin at their upper limit when folded.

This is a wiring/calibration result. Camera-driven learning and complete pickup
are assessed separately; direct stimulation is not presented as autonomous
image-based control. Evidence: `reports/motor-control/calibration/calibration.json`.

## Current measured outcome

The mapping passed 14/14 physical checks. The trained model scored 47/70 on
held-out joint responses at best, but **0/2 frozen full attempts and 0/1 online
attempt picked up and held the cube**. Read [RESULTS.md](reports/motor-control/RESULTS.md)
for the exact limits. The current model folder is:

`$HOME/code/codex/fly-brain/data/checkpoints/rebot/seven-motor-20260910-151942/seed-0`

## Run the stages

Generated data and checkpoints remain in this project’s data directory; downloaded references stay shared.

```sh
cd $HOME/code/codex/fly-brain
export MALECNS_HOME=$HOME/code/codex/fly-brain/data
STAMP=$(date +%Y%m%d-%H%M%S)

.venv/bin/fly-brain motor-prepare
.venv/bin/fly-brain motor-calibrate --output "reports/motor-calibration-$STAMP"
.venv/bin/fly-brain motor-collect \
  --output "$MALECNS_HOME/datasets/visual-dopamine/seven-motor-$STAMP"
.venv/bin/fly-brain motor-train \
  --dataset "$MALECNS_HOME/datasets/visual-dopamine/seven-motor-$STAMP" \
  --output "$MALECNS_HOME/checkpoints/rebot/seven-motor-$STAMP" \
  --trials 100000 --seeds 0 1 2
.venv/bin/fly-brain motor-run \
  --checkpoint "$MALECNS_HOME/checkpoints/rebot/seven-motor-$STAMP/seed-0" \
  --episodes 2 --max-steps 120 \
  --output "$MALECNS_HOME/runs/seven-motor-$STAMP"
```

An interrupted dataset can be continued with `motor-collect --output DATASET
--resume`. Completed scenes remain intact; incomplete scenes are retained with an
INVALID marker and excluded. Near-cube probes rebuild the contact world after
folding so prior collisions cannot displace the next trial's cube during reset.
The native simulator caches immutable collision shapes when rebuilding worlds.

`motor-run --axis 1` isolates J1; axes1–6 are joints and axis7 is the gripper.
Without `--axis`, all channels are available. `--selection strongest` tests only
the strongest nonzero available command each step; the default uses all channels.
`--learn` applies online PAM-gated updates and saves a new checkpoint in Results.
Otherwise weights remain fixed. The controller uses images for its decisions and
body state for bounded relative action execution. The reward evaluator alone sees
cube geometry and scores progress/success; it can terminate trials.

The UI recognizes seven-channel checkpoints through **Choose model…** and runs
`motor-run` when **Run fly brain** is pressed. Reset to folded for a new attempt.
**Learn from reward** explicitly enables exploration and checkpoint saving.

## Training and its limits

Collection starts each scene folded, then uses conventional IK only to establish
curriculum poses: folded, unfolding, approach, above-cube, and near-cube. Every
nonzero training reward comes from a physically measured directional action.
Training samples these recorded response outcomes; trial count is not a count of
new live simulations. IK targets, cube poses and stage labels are excluded from
actor inputs. A no-command choice receives neutral progress reward 0.5 by
definition. Reward for a directional action is 1 for improvement, 0 for worsening,
and 0.5 for no measurable change.

The neural model learns direction or no-command for each channel. Pose progress
combines grasp position and orientation; gripper scoring rewards opening during
approach and closing near an aligned cube. This initial reward bank does not
contain a learned lifting/holding curriculum. Full pickup/hold success is counted
only when the native physics evaluator actually confirms it, not when response
accuracy or grasp distance improves. See the measured run results before relying
on any checkpoint.

Export with `fly-brain export CHECKPOINT BUNDLE --include-graph`, or let other
projects reference the existing shared root. The bundle includes the parent
visual circuit and the new motor circuit. The original one-degree response
checkpoints and older state controllers remain readable.

Biological context: [leg premotor connectivity](https://www.nature.com/articles/s41586-024-07600-z)
and [leg motor-unit physiology](https://elifesciences.org/articles/56754).
