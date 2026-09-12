# Run the controller in the simulator

Open **ReBot Motion Lab Codex.app** in Finder from:

`~/github/rebot-motion-lab-codex/dist/`

Normal startup selects the configured checkpoint and the MLX runtime automatically. No shell script is required.

1. Click **Set up cube**. This starts a 20 mm task with the arm folded and gripper closed.
2. Keep the cube near **X = 350 mm, Y = 0 mm** for the currently deployed model.
3. Click **Run fly brain**, with **Learn from reward** unchecked.
4. For another attempt, click **Reset**, edit the cube placement if desired, and click **Apply cube** before running again.

The cube can be moved through the position fields or **Place with mouse** before applying it. All native floor and actuator limits remain enabled. The controller uses the fixed Front and Gripper observation cameras regardless of which viewport camera you select.

After the native evaluator confirms the five-second hold, the runner keeps the arm at the reached pose, opens the gripper fully, and verifies that the cube has landed on the floor before returning control to the UI. This final opening is an explicit task-completion command; the pickup and hold are neural. A failed or interrupted pickup does not trigger the drop step. **Results…** opens the run log, both camera recordings and native evaluation measurements.

## What was verified

- The selected controller passed **19/20** fresh native trials over **X=343–357 mm, Y=−10–8 mm**, with no runtime errors. This is only a small region around the starting cube, not general floor coverage.
- The same weights previously passed **15/18** on the preceding grid. Later fits were rejected; the selected controller is the retained baseline, not a claim that those fits improved learning.
- The real UI button passed **3/3** checks, including resetting and moving the cube to **[347, 8] mm** before another run.
- The window stayed open and accepted mouse events. The active, non-interactive 3D brain overlay was captured and inspected.
- A normal Finder-style launch, followed by **Set up cube → Run fly brain**, completed pickup, five-second hold and verified drop using MLX on September 12. Evidence: `reports/normal-ui-startup-20260912/normal-launch-test.json`.

These are experimental results over a small workspace, not evidence of reliable pickup at every floor position or cube size. Larger sideways moves, including Y≈58 mm, have failed. Wider-position retraining is in progress; the current selected checkpoint remains the older baseline.

The optional `scripts/open-trained-ui` launcher remains available for a separate managed window. Normal application startup reads `configs/trained-runtime.json` and preserves explicitly selected checkpoints. All generated models, data and reports remain under the local or Blacktower Codex fly-brain project directories.

Evidence: `reports/retain-grasp-20260911/UI_DEPLOYMENT.json`, `control-replay.html`, `ui-live.png`, and `final-tests.log`.
