> Historical experiment note. For the current published two-camera controller and model downloads, start with [README.md](README.md) and [training.md](docs/training.md). Referenced local reports/data are not all included in the public release.

# MaleCNS implementation and experiment results

Completed software implementation and initial validation on 2026-09-09.

Subsequent UI launch controls and shutdown fixes are documented in [UI_CONTROL.md](UI_CONTROL.md). The new UI can run against the cube already placed in the same window.

**The pipeline works, but a reliable autonomous fly-graph pickup policy has not been achieved.** The final correction-trained checkpoint completed 0 of 5 autonomous trials. All checkpoints are experimental and unqualified; none is presented as a ready pickup controller.

## Delivered

- Installable Python package and CLI with pinned dependencies (`uv.lock`).
- Shared asset verification/downloads, streaming connectome import, sparse graph caches, and provenance.
- Full selected-neuron recurrent controller with trainable sensory/motor adapters and neuron dynamics; optional fast/slow channels.
- State and image-only encoders, bounded joint/gripper commands, and optional recorded floor projection.
- Size-aware conventional teacher, synchronized recordings, data-quality audit, configuration-level splits, and DAgger corrections.
- Behaviour cloning, cached frozen-reservoir training, GRU baseline, graph/input ablations, full-size degree-preserving rewiring, and optional recurrent PPO.
- Independent hold/release evaluation, portable inference exports, and standalone core export for other robot adapters.

## Verification

| Check | Result |
|---|---|
| Raw data | All three files verified by full SHA-256; reused in place |
| Graph | 167,124 selected neurons; 25,578,853 directed weighted edges |
| Input/output populations | 17,883 sensory inputs; 2,129 outputs; all outputs reachable |
| Normalized CSR memory | 205,299,324 bytes |
| One-channel inference | p95 53.76 ms |
| Two-channel inference | p95 57.50 ms after vector-kernel optimization |
| Gradients | Numerical fixture checks and full-graph finite-gradient checks passed |
| Python | 32 tests passed |
| Native Swift | 32 RobotCore + 8 RobotControl tests passed; universal app rebuilt |
| MCP | 9 integration groups passed |
| Pilot | 50 successful conventional demonstrations in 54 attempts |
| Pilot integrity | 3,579 synchronized decision records; no audit errors |
| Initial split | 40 training, 5 validation, 5 test configurations |
| Final conventional boundaries | 10 mm and 90 mm cubes both picked up, held for 5 seconds, and released at (350, 0) mm |
| Final episode isolation | 3 clean fresh-process starts after recorded failed-policy poses |
| Sharing | Real exported checkpoint loaded and inferred from a separate project directory using the same prepared graph |

Four pilot attempts timed out during setup, before recording. Each placement succeeded in a fresh retry; failed attempts remain in the collection report. Successful pilot and final boundary checks met the requested 5 Hz collection budget. The final boundary-loop p95 values were 98.5 and 101.5 ms.

The selected neuronal set is explicit: traced cells plus classified null-status cells. Glia, orphan/anchor/assignment segments, unimportant segments, and unclassified null-status rows are excluded and counted. This does not treat all 151,856,684 raw segment-edge rows as complete neurons. Transmitter labels are retained as features without assuming a complete physiological sign/time-constant model.

## Learned-policy results

| Checkpoint | Method | Offline epochs | Autonomous pick-hold-release |
|---|---|---:|---:|
| `pilot-reservoir-seed0` | One-channel frozen core/readout | 100 | 0/5 |
| `pilot-state-bc-seed0` | Trainable adapters/dynamics, fixed graph | 2 | 0/5 |
| `pilot-fast-slow-reservoir-seed0` | Two-channel frozen core/readout | 200 | 0/5 |
| `corrected-fast-slow-seed0` | Warm-start readout with validated corrections | 300 | 0/5 |
| `gru-diagnostic-seed0` | Conventional recurrent baseline | 20 | 0/5 |

The first two runs used strict native endpoint rejection. Later runs used an explicitly logged floor projection that scales model proposals toward the current robot pose. It supplies no cube target or teacher action. Alignment, gripper timing, and closed-loop distribution shift remain failures. The correction-trained model's best offline validation loss was 0.0316, which did not translate into task success.

Twelve teacher-assisted correction attempts produced three completed tasks. The final two used cube-pose feedback to compensate for imperfect grasps and both completed. Older failed correction runs are retained but excluded from training because their recovery labels were not validated. The final training set contained 43 configurations; validation and test assignments were preserved.

These runs are diagnostics, with different optimization budgets. They establish neither a fly-topology advantage nor a fair final ranking. The five pilot test configurations were inspected during debugging; a future promotion test must use fresh independent configurations and repeated seeds. No 90/100 promotion test has passed. PPO and autonomous vision training remain gated research stages; no qualified-policy RL run was started.

## Simulator and runtime changes

Only the Codex simulator clone on `feature/fly-brain-codex` was modified; the original simulator checkout was left alone.

- Camera-object identity fixes replaced observation views, and reconfiguration reuses attached camera rigs.
- Episode counters reset while RealityKit's internal timebase stays monotonic.
- A controller-scoped macOS activity prevents background throttling; it ends with ownership.
- Learned-policy evaluation, DAgger, and PPO use fresh owned simulator processes by default to avoid carrying physics state out of extreme failed poses. Conventional collection reuses its validated scene.
- Locked-console preflight, stalled-frame rejection, and temporary display/system idle-sleep assertions keep live collection explicit and bounded. No screen unlocking or power-preference changes occur.

Reusing an explicitly supplied existing simulator still exposes RealityKit's in-process reset limitations after extreme poses. The final fresh-process check passed. Physics is real time, not bitwise deterministic; accelerated/batched physics, a MuJoCo port, and real hardware control are not implemented.

## Locations and evidence

- Shared root: `$HOME/code/data/malecns/`
- Pilot/corrections: `datasets/rebot-pick/pilot-v1/`
- Learned checkpoints: `checkpoints/rebot/`
- Portable inference export: `exports/corrected-fast-slow-seed0/`
- Learned core export: `checkpoints/core/pilot-state-bc-seed0/`
- Original graph ID: `53e1d528e43cb834982e9207457da777d4c0d55f570844bb96b049ad043c3f7d`
- Rewired control ID: `e503cad174b84a78d91dbec6100eee2daea31315d6c30a7011f9ba4a021ac908`

The control graph completed 25,578,853 directed double-edge swaps and preserved both degree sequences and the weight multiset. A trained rewired-policy comparison has not been run.

See `reports/model-registry.json`, individual evaluation directories, `reports/pilot-dataset-audit.json`, `reports/final-native-check.json`, `reports/cross-project-load.json`, benchmark reports, and `README.md` for commands and reproducibility details. All data/model paths are outside the code repository and reusable across projects.

## 20 mm folded-start collection fix

Simulator 1.8 uses adjacent convex collision sections clipped from the original fingertip meshes and calibrates the closing stop to the inner finger surfaces. Ten of ten 20 mm teacher demonstrations passed from folded starts; the audit and training loader passed. External controller floor contact now limits downward travel while allowing upward recovery in the same session. See [TRAIN_CUBE20_FOLDED.md](TRAIN_CUBE20_FOLDED.md) and `reports/teacher-fix-verification/`. These are teacher/data checks; the existing neural checkpoints have not been retrained or requalified.

The verified collection recipe is for 20 mm cubes in X=335–365 mm, Y=-15–15 mm, yaw=0, starting folded. A separate 50 mm folded-start compatibility trial passed. The separate 10 mm and 90 mm folded-start trials failed, so those sizes are not qualified by this fix. Cube placement/resizing still supports the existing 10–90 mm range.

## 2026-09-10: retinal input and dopamine-gated response learning

A separate `visual-*` controller now routes RGB samples through identified R1–R6,
visual KCg-d, MBON, descending and leg motor populations. Actual PAM01-gated
plasticity changes only existing KC→MBON edges. The tested task is one degree of
base rotation toward a 20 mm cube from folded; full pickup remains unfinished.

The final model passed 16/16 fresh single-turn simulator trials. Three recorded
held-out comparisons reached 8/8 with dopamine, 0/8 without dopamine, and 4/8 with
visual input blocked. The harder repeated-turn task remains unreliable. UI learning
creates a separate checkpoint; source weights are preserved. See
[full results](reports/visual-dopamine/RESULTS.md) and
[run instructions](RETINAL_DOPAMINE.md).

## Seven-channel motor mapping and measured training

The follow-on request mapped 75 motor neurons to six joints and the gripper and
added the intervening CB/descending/VNC circuitry. Fourteen direct-stimulation
checks passed. Thirty camera scenes with 420 physical directional probes trained
three seeds plus dopamine-disabled controls. Best held-out response accuracy was
47/70. Continuous pickup/hold failed in two fixed-policy runs and one online run.

The UI, shared checkpoints, portable exports and reward-gated learning were
verified; successful autonomous pickup remains unfinished. See
[MOTOR_NEURON_MAP.md](MOTOR_NEURON_MAP.md) and the
[full evidence report](reports/motor-control/RESULTS.md).
