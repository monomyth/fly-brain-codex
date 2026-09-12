# Camera views and Orbbec RGB optics

The approved camera placement is retained. The simulator offers **Orbit, Front, Top, and Gripper**, with synchronized Front/Gripper RGB observations. Front now models the Gemini 336L RGB FOV; Gripper models Gemini 305 RGB FOV. No collection or training was run during this optics update.

## Front placement

Robot-base coordinates use +X toward the nominal cube, +Z upward, and meters in camera calibration.

| Setting | Value |
|---|---|
| Camera center | X 750, Y 0, Z 650 mm |
| Fixed look-at point | X 150, Y 0, Z 50 mm |
| Downward angle | 45 degrees |
| Nominal RGB field of view | 94 degrees horizontal x 68 degrees vertical |
| Recorded image | 320 x 200 RGB, 16:10 |

This puts the camera beyond the cube, facing back toward the robot, so the cube is in the foreground and the folded arm is behind it. The camera is lower than the default Orbit camera (about 1,057 mm above the floor). Framing includes the complete folded arm and the nominal pickup area with a border margin. It does not track the cube or consume the cube's true position. Arbitrary points across the entire 4 m floor are not all visible or reachable.

The engineering choices follow the general practices of keeping the work area in view, avoiding self-occlusion, and choosing the closest useful working distance with a margin at the image edges. The exact 45-degree angle comes from your request; it is not a universal robotics standard. Sources: [Zivid camera positioning](https://support.zivid.com/en/latest/camera/academy/camera/capturing-high-quality-point-clouds/working-distance-and-camera-positioning.html) and [robot-mounted camera clearance](https://support.zivid.com/en/latest/camera/getting-started/user-guide/mechanical-installation/available-mounts/robot-mounting.html).

Both devices specify **RGB H94 x V68 degrees (±3 degrees)** at 1280 x 800, or the same field when uniformly downsampled to the simulation's 320 x 200. Sources: [Gemini 336L official specifications](https://store.orbbec.com/products/gemini-336l), [Gemini 305 official specifications](https://www.orbbec.com/gemini-305/). These are the RGB specifications; depth has a different FOV and is not simulated here.

The renderer uses a square-pixel ideal pinhole with vertical FOV 68 degrees and aspect 16:10, giving horizontal FOV 94.3636 degrees. Calibration reports that actual value alongside the nominal 94 x 68 specification. Front/Gripper views are letterboxed to 16:10 in the main window, so window resizing does not change the sensor FOV. Their small observation previews use the same aspect. Orbit and Top retain their prior behavior.

## Gripper: supplied Gemini 305 cradle assembly

The current mount and housing are the meshes from the supplied [assembly preview](/Users/monomyth/github/rebot-b601-trinkets/output/cradle_support/B601_Gemini305_CradleSupportPreview.FCStd), registered to the simulator's `end_link` frame. The source project is read-only. The latest [side-view reference](/Users/monomyth/github/rebot-b601-trinkets/output/cradle_support/assembly_side.png) and its assembly source specify **15 degrees downward**, which supersedes the earlier 45-degree Gripper approximation. Front stays at 45 degrees.

| Setting | Value |
|---|---|
| Camera rear midpoint, relative to `end_link` | X -88.2093, Y 0, Z +66 mm |
| Attachment | Actual cradle mesh and support leg from the reference assembly |
| Downward optical angle relative to gripper | 15 degrees |
| Nominal front optical midpoint | X -65.9930, Y 0, Z +60.0472 mm |
| Color stream used | Left RGB, 9 mm left of the optical midpoint |
| Simulated vertical field of view | 68 degrees |
| Recorded image | 320 x 200 RGB, 16:10 |
| Ideal-pinhole horizontal field of view | About 94.4 degrees |

The CAD coordinate transform is tool X = CAD Z - 103.2093277 mm, tool Y = CAD X, tool Z = CAD Y. This maps the CAD fingertip to the tool origin. Alignment against 5,877 vertices of the existing gripper base has a median nearest-vertex distance of 0.0493 mm; tessellation and some source parts differ. This is a geometric registration check, not a claim of exact identity of every gripper component.

The complete supplied camera and cradle shape are rendered. The optical center remains an ideal pinhole approximation at the nominal 23 mm front plane, with an 18 mm stereo baseline used to select one RGB viewpoint. Actual factory intrinsics, lens distortion, entrance-pupil position, stereo depth, camera mass, and bracket collisions are not simulated. The view follows the wrist without object tracking or world-level stabilization. Any gripper occlusion or object outside its field of view remains visible in the recorded data; the mount is not moved to expose the cube.

Canonical converted meshes and source checksums are in `data/camera-rigs/cradle-v3/`. The simulator bundles copies under `Sources/RobotCore/Resources/model/camera/`. The conversion and registration check are reproducible with `scripts/prepare_cradle_camera.py` after extracting the three source meshes with FreeCAD. Device RGB specifications are from [Orbbec](https://www.orbbec.com/gemini-305/).

## Collection and model compatibility

- `rebot_get_observation(images=true)` now returns **Front + Gripper**, both from one frozen scene state. Top remains a selectable overview, and Orbit remains the movable spectator camera.
- Every frame includes its current `world_from_camera`, intrinsics, dimensions, mount transform, camera model/stream, and rig revision `front336l-gripper305-rgb-v4`. This is essential for the moving Gripper camera.
- Episode manifests store the camera names and revision. The dataset writer and audit reject inconsistent or mixed rigs. Historical Front/Top episodes remain readable offline.
- The feedback trainer registers Front to the left retinal population and Gripper to the right. This is an engineered sensory mapping. The registration is saved in each checkpoint and verified at inference.
- Changing cameras requires new demonstrations and refitting retinal normalization. Camera-aware warm starts preserve learned neural parameters while refitting the sensor adaptation; old and new camera datasets cannot silently mix.
- Old vision checkpoints are blocked from running against this new rig. All generated datasets, checkpoints, and review artifacts remain inside this project's directory.

The next collection uses new 20 mm folded-start demonstrations from this pair. Train and validate a matching checkpoint before selecting it in the UI. No claim of successful pickup with the new cameras has been made.

## Review and checks

The sidebar uses one explicit gray selected-row color in both focused and unfocused windows. Manual joint/slider pose updates remain immediate. The camera housing is render-only.

Open the persistent review window with:

```sh
cd /Users/monomyth/code/codex/fly-brain
.venv/bin/python scripts/preview_camera_rig.py
```

This command saves a few diagnostic images and leaves the simulator open; it does not create a training dataset or start training. If the review window is already running, the command reuses it.

`reports/camera-fov-review/verification.json` records live camera attachment and synchronization checks. `image-calibration.json` checks Front projection against the visible cube pixels. Build and regression-test logs are saved alongside them.
