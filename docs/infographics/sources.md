# Infographic evidence

These diagrams describe the September 12, 2026 implementation, including its limitations. They are explanatory illustrations, not biological measurements.

- Runtime: `src/fly_brain/visual_dopamine/motor_policy.py`, `motor_run.py`, `motor_release.py`, `scripts/mlx_sensory_backend.py`, `scripts/probability_gripper.py`.
- Training: `scripts/head_training_runtime.py`, `scripts/continue_head_training.py`, checkpoint manifests and frozen targets in the [model release](https://huggingface.co/monomyth/fly-brain-codex).
- Native mechanics and cameras: [simulator branch](https://github.com/monomyth/rebot-motion-lab/tree/fly-brain-codex), `ObservationRig.swift`, `ManipulationScene.swift` and `ExperimentCoordinator.swift`.
- Trial results: [results and limitations](../results.md), with the baseline hold-only metric separated from the newer full-task metric.
- Biological source and attribution: [MaleCNS project](https://male-cns.janelia.org/download/).

The graph subset counts and population mappings belong to the declared preparation recipe; they are not counts of all cells in the raw segmentation. The camera and body encodings, muscle-pool mapping, contact-bank selector and completion release are engineering choices. The dopamine multiplier used in fitting does not establish biological reinforcement learning.
