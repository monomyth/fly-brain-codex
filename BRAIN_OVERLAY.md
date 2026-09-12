# Live 3D brain overlay

The Codex simulator shows a fixed, non-interactive MaleCNS soma point cloud in the top-right of the robot viewport, below its camera controls. Mouse gestures pass through to the robot viewport. The view includes the brain and ventral nerve cord, in a fixed oblique perspective.

## Run it

Open `~/github/rebot-motion-lab-codex/dist/ReBot Motion Lab Codex.app`, apply or place a cube, wait for Ready, and click **Run fly brain**. The overlay changes from Idle to Live when current neural activity arrives. **Stop fly brain** clears the activity and leaves the anatomy dim. Moving/resizing the cube starts a new episode; activity from the previous episode cannot appear as live.

The same overlay works with `fly-brain run` and `evaluate`, including a connected app via `--current-episode`. It shows activity from MaleCNS policies only. A conventional GRU, a teacher, or manually moving the robot does not generate fly-neuron activity.

## Downloads and shared storage

**No additional downloads or training are required.** The existing annotations contain soma locations. Derived geometry is generated automatically when running a MaleCNS evaluation, or can be prepared ahead of time:

```sh
cd /Users/monomyth/code/codex/fly-brain
.venv/bin/fly-brain --home /Users/monomyth/code/data/malecns prepare-overlay
```

For a different graph, add `--graph-id GRAPH_ID`. The shared cache is `MALECNS_HOME/visualizations/GRAPH_ID/soma-v1/`, outside both source repositories. It contains normalized 3D positions, the exact corresponding body IDs and controller indices, checksums, transformation metadata, and CC BY 4.0 attribution. Other projects can reuse it. Original graph files and checkpoints are unchanged.

The current graph has 167,124 neurons; 140,025 have finite soma locations. The 27,099 without locations are omitted from the geometry, and the displayed count makes that coverage explicit. Positions are joined by body ID, never by an assumed table order. The geometry is centered and uniformly scaled, with source axes x, -z, y mapped to display coordinates. No anatomical locations are invented.

Detailed neuron branches or a neuropil surface would require optional skeleton/mesh assets from https://male-cns.janelia.org/download/. They are not used by this version.

## What the color means

Blue brightness represents the mean absolute value of a neuron's current continuous state across the model's channels, quantized to 8 bits for display. A fixed logarithmic display gain, `log(1 + 999 * value) / log(1000)`, makes small activations visible. This does not change the controller or the raw mean/peak numbers shown below the brain. For the current checkpoint the activation averages the fast and slow channels.

Gold dots mark changes in that magnitude since the preceding displayed update. The strongest 4,000 changes are highlighted through the volume so quiet foreground neurons cannot hide them; the counter includes all changing mapped neurons, including those outside the displayed strongest subset. Gold is a change marker, not a spike or a claim of excitation. No highlights are generated for unchanged values, a first frame, a different episode/graph, or a gap over 1.5 seconds. The simulation-time readout advances with incoming observations. The display is simulated controller activation, not biological spikes, a firing rate in Hz, or a causal explanation for an arm movement.

Frames carry the episode ID, observation frame ID, simulation time and graph ID. A frame describes the neural update from that observation whose proposed action was accepted by the simulator (possibly with floor limiting). The overlay is refreshed at up to 5 Hz; it does not invent activity between controller steps. Frames older than 1.5 seconds, from another episode, or without a MaleCNS controller owner are not shown as live. Individual soma positions can overlap in projection; the gold layer is intentionally visible through the anatomical point cloud. Mouse gestures still pass through the entire overlay.

## Implementation and verification

Python writes one latest-frame file, `brain-activity.json`, atomically in that simulator's private control directory. A background worker keeps at most one pending frame. The native app reads and validates it off the main thread and renders the point cloud with SceneKit. There are no added motor commands or round trips through the control socket. Display failures warn and leave robot control operational. Geometry size, SHA-256, finite positions, graph IDs, payload length and frame freshness are validated. Activity is not currently saved as a replay recording.

Verified in `reports/brain-overlay-verification/result.json`: native UI Run/Stop with 40 mm and 60 mm cubes; live changing neuronal values; current episode matching; placement preserved; Stop clears activity. Screenshots are `live.png` and `idle.png`. Both Python and native test suites pass, including body-ID alignment, malformed display data, stale/foreign episodes, and display I/O failure isolation. These checks validate the overlay, not autonomous pickup success.

### Visibility correction

The initial linear color scale made real activity hard to see: in an inspected run, 99% of displayed values were below 0.08. The enhanced display adds fixed logarithmic contrast, gold change markers, the changing-neuron count, raw mean/peak activity, and simulation time. The model, telemetry values and motor commands are unchanged. Verification artifacts are in `reports/brain-visibility-verification/`.
