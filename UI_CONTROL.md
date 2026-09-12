# Run the fly-brain controller from the simulator

Open the rebuilt `ReBot Motion Lab Codex.app` in
`/Users/monomyth/github/rebot-motion-lab-codex/dist/`.

1. In **Cube pickup experiment**, choose **Set up cube** if needed.
2. Set **Cube X mm**, **Cube Y mm**, **Cube side mm**, and yaw, then choose **Apply cube**. Alternatively choose **Place with mouse** and click the floor.
3. Wait for **Ready**, then choose **Run fly brain**.
4. **Stop fly brain** stops the policy and arm, leaving this window open.
5. Apply the next cube placement or use **Reset**, then run again.

The UI remembers the last chosen checkpoint. With no saved choice or environment override, the default is `corrected-fast-slow-seed0`. **Choose model…** selects
another MaleCNS checkpoint folder. The input mode is taken from the selected
checkpoint and displayed in the panel; the current checkpoint uses simulator
state. Starting preserves the cube position, size, robot pose, and episode ID.
The model's existing reliability limits still apply: this starts an experimental
attempt, not a qualified pickup skill.

**Start physics** runs the simulation without starting a policy. Cube edits are
locked while the model loads or controls the arm. Stop or reset cancels the owned
policy process. **Results…** opens the run folder containing `runner.log`,
`evaluation.json`, and recorded observations/images.

UI results are stored under `MALECNS_HOME/runs/rebot-ui/`. Defaults use
`~/code/codex/fly-brain/data` and `~/code/codex/fly-brain/.venv/bin/fly-brain`.
Advanced installations may set `MALECNS_HOME`, `FLY_BRAIN_CHECKPOINT`, or
`FLY_BRAIN_EXECUTABLE` before launching the app.

## Retinal response controller

For camera pixels and PAM01-gated learning, follow [RETINAL_DOPAMINE.md](RETINAL_DOPAMINE.md).
The panel identifies this as a base-turn experiment. **Learn from reward** enables
exploration and saves a new checkpoint in shared Results; it is off for fixed-model
evaluation. These controls do not turn the response checkpoint into a pickup skill.

## Terminal behavior and fixes

Ordinary Terminal runs still own temporary simulator instances and close those
instances when the run ends. They now launch with `--external-controller`, which
automatically enables local control while accepting normal mouse input.
Previously `--mcp-integration-test` made the window ignore mouse events, allowing
clicks to reach other apps underneath it.

The UI uses the attach mode below, which never resets or launches another app:

```sh
.venv/bin/fly-brain run \
  --current-episode \
  --control-directory /path/to/current/private/control-directory \
  --checkpoint /path/to/MaleCNS/checkpoint \
  --record-images --floor-projection \
  --output /path/to/new/run-results
```

An optional `--episode-id` rejects stale launches after the cube episode changes.
The selected scene must already be Ready and use the checkpoint's input mode.
The UI handles those requirements directly.

Cleanup now preserves the original failure, records any secondary release error,
and finishes the report if the app disconnects. Ctrl+C exits with status 130 and
saves an interrupted result. Owned native and power-guard processes have separate
process groups so Terminal interrupts reach the runner first.

## Verification

- Native accessibility button actions started and stopped real model processes.
- Two placements, `[340, -20]` with a 40 mm cube and `[360, 20]` with a 60 mm cube,
  preserved their episode IDs and configuration while issuing neural actions.
- The simulator stayed open after Stop.
- A physical click on the main window title bar kept that app foreground.
- Ctrl+C and deliberately closing an owned simulator saved reports without a
  cleanup traceback.
- Evidence: `reports/ui-run-verification/` and `reports/shutdown-verification/`.

## Live brain overlay

A non-interactive 3D MaleCNS activity view now sits at the top-right of the robot viewport. Use **Run fly brain** after placing a cube; the view updates automatically and dims on Stop. No extra downloads are needed. See [BRAIN_OVERLAY.md](BRAIN_OVERLAY.md) for shared geometry, color meaning, and verification.

## Task imports

The toolbar imports cube tasks as well as motion trajectories. **Load task…** is always visible in the cube panel and is enabled when it is safe to replace the task. Task lists for CLI collection are not individual UI task documents. Invalid task fields now identify the failing field instead of showing a generic format error.

## Seven-channel motor checkpoints

Choose a checkpoint from the [motor-neuron experiment](MOTOR_NEURON_MAP.md) to
run all six joints and the gripper. The brain overlay displays seven motor drives
and the PAM01 reward value. Run dispatches to `motor-run`; Stop leaves the window
open. The current checkpoint has not completed pickup in live testing. Its UI
reports grasp distance and an incomplete pickup rather than claiming success.
