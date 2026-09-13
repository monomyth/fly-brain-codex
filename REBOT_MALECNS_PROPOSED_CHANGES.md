# ReBot Motion Lab: proposed support for a MaleCNS pickup experiment

Proposal date: 2026-09-09. Based on read-only inspection of `$HOME/github/rebot-motion-lab`. No changes to that repository have been made. This document proposes future work; it does not authorize implementation.

## Objective and scope

Enable an external controller built around the MaleCNS connectome to observe the simulated arm and a cube, approach and grasp the cube, lift it, and keep it level. Interpret “parallel to the ground” as the cube's top and bottom faces remaining horizontal; specify a gripper orientation separately if required.

Implement the first version in the native macOS app. It already exposes arm, gripper, camera, and state controls through MCP. Browser parity is a later milestone. Run the neural model, dataset loading, and training in a separate process, such as one in the `fly-brain` project.

The simulator's readiness and the learned controller's success are separate acceptance decisions. A working simulation interface does not establish that MaleCNS can learn this task.

## Existing capabilities to reuse

- Six independently commanded arm joints and a coupled gripper with a nominal 0–90 mm opening.
- Forward kinematics, position-only inverse kinematics, and interpolated motion playback.
- Geometric floor protection covering all moving links and both fingertips. The existing floor surface is at base-frame Z = −1 mm.
- Native `Front`, `Top`, and `Orbit` views, including MCP camera selection.
- Same-user Unix socket IPC and a native stdio MCP server.

Current gaps: no manipulated objects, grasp physics, object observations, image observations, orientation-constrained IK, or episodic experiment interface. Existing MCP pose commands require the previous motion to finish or be stopped.

## Proposed changes, in implementation order

### 1. Add a configurable cube and task configuration

Add an identified cube object with configurable size, pose, mass, and contact material. Choose an initial size comfortably inside the gripper's effective aperture; verify the actual finger geometry rather than relying only on the nominal opening value.

Place the cube on the actual floor surface. For an upright cube, its center height is the floor height plus half its side length. Validate robot clearance and choose a reachable initial placement. Distinguish an invalid configuration from an unsuccessful control attempt.

Persist task configuration separately from the existing trajectory format. Include target clearance, allowable tilt, hold duration, and initial robot pose. A face marker can make orientation easier to see without changing the physics.

Acceptance: the cube has the requested dimensions and pose, rests on the floor without overlap, and resets to the configured state.

### 2. Implement object contact and grasp physics

Add a static floor collider, a dynamic rigid cube, and collision geometry for moving robot links and both gripper fingers. Couple the robot's commanded motion to its collision representation; prevent fingers from passing through the cube. Make the accepted gripper aperture reflect contact, rather than always reaching an impossible requested opening.

First run a focused RealityKit feasibility check: resting contact, pushing, bilateral pinch, slow lift, and release. Establish appropriate collision geometry, motion limits, and physics update handling before adding learning. Robot links may remain position-controlled; this does not require claiming that motor torques or actuator dynamics are simulated.

For the primary experiment, the cube must be supported by contact under gravity. If an explicit attachment constraint is used for an initial demonstration, expose and log it as a separate grasp mode. Do not silently attach the cube when a close command is issued.

Acceptance: missed grasps fail; successful grasps lift; opening the fingers releases the cube; unsupported cubes fall. Verify that kinematic finger motion does not inject unrealistic velocities or tunnel through the object.

### 3. Add orientation-aware tool control

Define documented transforms for the world, robot base, tool, grasp center, and cube. Use named coordinate frames and explicit units. Provide position plus quaternion targets, including a documented quaternion component order, and return actual tool orientation.

Extend IK to consider position and orientation, or provide an equivalent external solver interface using the existing full joint commands. Support a level-hold constraint and report both position and angular error. Check the motion path for interference with the floor and objects; a valid endpoint alone is insufficient.

Acceptance: the gripper follows a reachable lift while maintaining its requested orientation, and unreachable or obstructed commands report the actual accepted outcome.

### 4. Add observation cameras and image capture

Reuse the existing front and top viewing geometry. Add named observation cameras that can be calibrated and locked independently of the spectator's orbit/pan controls. Support capturing both views for the same observation; simultaneous onscreen viewports are optional.

Return RGB frames with image dimensions, camera intrinsics/extrinsics, coordinate conventions, episode ID, frame ID, and simulation timestamp. Associate joint and object telemetry with those frames, or explicitly report capture skew. Keep UI overlays out of policy images.

Depth and segmentation are optional later channels, subject to backend support. Declare which channels the controller receives. Displaying a camera view alone is not an image interface for MaleCNS.

Acceptance: frame-to-world calibration is checked against known scene points, the cube remains visible in the configured workspace, and spectator camera changes do not alter policy observations.

### 5. Expose manipulation observations

Provide actual joint angles, gripper aperture, tool pose, timestamps, per-finger contact state, and motion status. Include velocities where measured or derived with a documented time interval. Expose contact forces only if the backend actually provides them.

Maintain evaluator telemetry for cube pose, clearance, tilt, motion, and floor contact. Derive grasp/hold status from observed support and stability, not from whether the controller requested “close.”

Support explicit input modes: state-assisted debugging and vision-based evaluation. Exact cube coordinates, segmentation, and evaluator success signals must not silently enter a run labelled vision-based. In training, reward access should be declared separately from policy observations.

Acceptance: requested actions, executed motion, contact, and success are independently observable; a closed empty gripper is not reported as holding the cube.

### 6. Add an external-controller session

Introduce an explicit control mode with one active motion owner: manual controls, sequence playback, or an external controller. The external process submits bounded joint targets or tool-pose increments plus gripper commands; the simulator executes them through its local motion and contact logic.

Support frequent updates without repeatedly cancelling ordinary preset playback. Keep the neural process outside the render thread. Reject stale episode/frame IDs and return action acknowledgements identifying requested and executed results. Manual takeover, explicit stop, and controller disconnection must have documented behaviour.

Retain the private same-user transport. The current IPC limit is 1 MiB, so image transport needs an explicit bounded design, such as compressed frames or a local shared-memory channel. Do not assume MCP chat calls are the per-frame control loop.

Acceptance: sustained observation/action exchange works at a measured rate, the UI remains responsive, conflicting writers cannot command the arm, and disconnect stops further commanded motion. Stopping arm commands does not secretly freeze cube physics.

### 7. Add episode lifecycle and clock semantics

Add configure, reset, start, pause/resume, stop, and observe operations for experiments. Reset the robot, cube, velocities, contacts, pending actions, and success counters as one transition. Expose when reset and initial settling have completed. Notify the external controller to reset its neural state independently.

Separate experiment time from display refresh and ordinary waypoint playback. Physics must continue while the controller holds a stationary target. Define pausing the entire simulation separately from stopping only the arm.

Start with timestamped real-time episodes if that is what the chosen backend supports. Advertise a fixed-step action API only after verifying that physics and observations can advance in controlled steps. Seed placement and noise, record runtime settings, and measure repeatability; do not promise bitwise deterministic RealityKit physics.

Acceptance: resets leave no state from the previous episode, no partial-reset observations escape, and timing remains meaningful during stationary holds and slow controller inference.

### 8. Add task scoring and evaluation

Make success criteria configurable. Suggested initial criteria are cube-bottom clearance of at least 100 mm above the floor, top-face tilt no more than 5 degrees, and stable support by the gripper for five consecutive simulated seconds. Define numerical stability tolerances before evaluation.

Exclude transient upward motion, cube support by another surface, and paused time from a successful hold. Reset the hold timer if any condition fails. Record timeout, drop, lost grasp, workspace violation, and prohibited collision separately.

Use a configured initial cube face to measure tilt; a marked face avoids visual ambiguity from cube symmetry. Training reward terms may include approach, alignment, grasp, lift, and stability, but evaluation must use the independent success criteria.

Acceptance: scripted negative cases cannot pass; repeat successful trials across a recorded set of reachable cube placements and report the success rate.

### 9. Record demonstrations and controller runs

Add a recorder for timestamped images, observations, proposed actions, accepted actions, and outcomes. Save task configuration, random seed, simulator version, camera calibration, physics settings, and declared observation channels.

Allow manual or conventional-controller demonstrations to be recorded for imitation training. Track control provenance: manual, conventional, teacher-assisted, or MaleCNS-based. Let the external controller supply its model/checkpoint ID and connectome preprocessing identifier for the run log.

Separate visual playback of recorded data from re-running a controller or re-simulating physics. Report divergence tolerances for the latter.

Acceptance: an exported run can be inspected without the original live session, and evaluation runs identify whether demonstration guidance was present.

### 10. Extend MCP orchestration and the experiment UI

Expose setup and inspection through additive MCP tools. Illustrative names: `rebot_configure_task`, `rebot_reset_episode`, `rebot_get_observation`, `rebot_move_to_pose`, and `rebot_experiment_control`. Final schemas should follow the observation and lifecycle contracts above.

Keep existing camera tools for Grok or another configured MCP client. Add task status, cube settings, control-owner indication, observation previews, contact indicators, clearance/tilt/hold measurements, and visible stop/reset controls.

Document which interface orchestrates experiments and which carries frequent policy actions. Preserve existing tools and trajectory import/export behaviour outside experiment mode.

Acceptance: an MCP client can configure a trial, establish views, start an external-controller session, inspect results, and reset without editing application code.

### 11. Preserve motion behaviour and add focused validation

Follow the repository's `AGENTS.md` and `HANDOFF.md`: manual sliders must update the permitted pose immediately, both fingertips must respect the floor, and presets/sequences must retain their existing interpolation. Do not reintroduce the old manual-motion filter or per-tick SwiftUI invalidation.

Add tests for pose orientation, cube placement, contact/release, reset isolation, observation/action sequencing, stale commands, ownership/disconnect, and scoring. Add native visual integration checks for contact and level holding, plus performance measurements with physics and image capture enabled. Use real scene observations to validate grasp behaviour; unit tests alone cannot establish it.

Acceptance: existing core/MCP checks still pass, immediate slider tracking remains intact, and recorded native trials demonstrate a conventional controller can pick up and hold the cube before evaluating MaleCNS.

## Expected code areas

Paths below are relative to `$HOME/github/rebot-motion-lab`; new filenames are suggestions.

| Area | Likely responsibility |
| --- | --- |
| `Sources/RobotCore/` | Task/observation/action types, frame transforms, orientation IK, scoring, experiment log formats |
| `Sources/ReBotMotionLab/RobotViewport.swift` and new scene helpers | Cube rendering, physics integration, colliders, observation cameras, capture |
| `Sources/ReBotMotionLab/AppModel.swift` and new experiment coordinator | Control ownership, lifecycle, reset, experiment timing |
| `Sources/ReBotMotionLab/MCPControl.swift` | Task orchestration and observation access |
| `Sources/RobotControl/` | Additive schemas, validation, transport for external-controller sessions |
| `Sources/ReBotMotionLab/SimulatorView.swift` or a dedicated experiment view | Task controls and feedback without per-frame UI rebuilds |
| `Tests/`, `Verification/`, `scripts/` | Contract tests and isolated native manipulation checks |
| `MCP.md`, `README.md`, `docs/DEVELOPMENT.md` | New capabilities, units, timing, input modes, grasp limitations, validation instructions |

## Delivery milestones

1. **Manipulation baseline:** cube, contacts, orientation control, and scoring; demonstrate a conventional pickup, lift, hold, and release.
2. **Controller-ready environment:** image/state observations, external control, reset, and recording; demonstrate the task through the external interface.
3. **MaleCNS experiment:** load and train the neural controller externally, remove teacher guidance for evaluation, and compare against conventional and neural-ablation controls.
4. **Optional expansion:** depth/segmentation, accelerated or batch training with a verified stepping backend, multiple objects, and browser parity.

The simulator repository owns the environment and measurable task contract. The external project owns MaleCNS data loading, neuron dynamics, input encoding, action decoding, training, and experiments that measure the connectome's contribution. Success in milestone 2 is a prerequisite for testing milestone 3, not proof that milestone 3 has succeeded.
