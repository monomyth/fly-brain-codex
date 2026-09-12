# Measured results and limitations

Snapshot: September 12, 2026. The profiles are distinct and their success metrics differ.

- **Retained baseline:** 19/20 native pickup-and-hold trials in X343–357 / Y−10–8 mm, with a 20 mm cube and folded start. This historical batch did not verify release. Later normal UI runs separately verified the explicit pickup/hold/drop sequence.
- **Newer touch-gated controller:** 12/15 complete pickup/hold/drop tasks. Twenty were planned; testing stopped after three failures, leaving five unrun. It did not pass the UI deployment gate.
- **Newest CUDA fit:** 12,000 iterations completed on an RTX 4090; not native-tested or deployed. Its fit metrics cannot be quoted as robot success.

The broader evaluation included a missed grasp at [345, −30] mm, a held cube exceeding the 5° tilt limit at [350, −30] mm, and a lost/tipped grasp at [355, +30] mm. Broader arbitrary placement and other sizes remain unproven.

A qualifying hold requires ≥100 mm floor clearance, ≤5° tilt and ≥5 seconds of continuous native stability. The later full-task evaluator also requires opening and verified floor landing. Release is an explicit completion routine, not a learned neural action. Displayed brain activity is activity of our engineered rate model, not measurements from a living fly.

Aggregate native records: [baseline](evidence/stable.json), [experimental](evidence/experimental.json). Full portable historical qualification/runtime records are in the corresponding [HF model bundles](https://huggingface.co/monomyth/fly-brain-codex/tree/main). A selected successful recording in the README is an illustration, not a representative rate estimate.
