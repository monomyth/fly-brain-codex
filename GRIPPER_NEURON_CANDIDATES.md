# Gripper-neuron candidates

Recorded 2026-09-11 at the user’s request. The controlled LTM variant is now implemented and passed 14/14 direct output checks. The matched learned comparison is complete: LTM passed 1/4 isolated grasp cases and 2/2 lift/hold cases, matching the static baseline. It is not promoted or established as an improvement.

## Candidates and evidence

- **MN7: opening analogy.** Activating these proboscis motor neurons spreads the two labella. MaleCNS examples: left body IDs 16134 and 16892; right 17495 and 32092. The 2020 functional study supports MN7 as the labellar abductor. Earlier work assigned spreading to MN8, so use the later evidence and inspect anatomical matching before exporting a circuit.
- **Long-tendon muscle (LTM) motor neurons: gripping analogy.** These operate the tarsal claw through the retractor unguis tendon and are relevant to substrate grip. MaleCNS labels include `ltm1-tibia MN`, `ltm2-femur MN`, and `ltm MN`. Left-front examples include 818553 and 822047 (ltm1), and 805309 and 1050615112 (ltm2).
- MN7 and LTM belong to separate biological systems; they are not an antagonistic opening/closing pair. I did not establish a dedicated labellar-closing counterpart to MN7 from the checked studies. MN1 retracts the proboscis and MN6 extends the labella; neither should be described as a verified gripper-closing neuron.

## Controlled experiment

Investigate a recorded LTM premotor circuit as a candidate for the robot’s grip channel. Mapping stronger activity to close/hold and weaker activity to release/open would be an explicit engineering adapter, not a claim that a biological antagonist has been identified. Compare it with the existing tarsal levator/depressor readout using the same cameras, cube tasks and physical success thresholds. Keep a new circuit identity and new checkpoints; do not overwrite existing mappings. The first matched comparison did not improve the isolated grasp result.

The temporal candidate retained the existing mapping and passed 2/4 isolated grasp cases, then 4/20 full CPU trials. The LTM comparison changed the grip endpoints separately. This small comparison does not establish the best biological analogue: the new LTM edges started from measured weights while shared previously learned weights were transferred.

## Sources and local verification

- McKellar et al. (2020), *Controlling motor neurons of every muscle for fly proboscis reaching*: https://elifesciences.org/articles/54978
- Azevedo et al. (2024), *Connectomic reconstruction of a female Drosophila ventral nerve cord*: https://doi.org/10.1038/s41586-024-07389-x
- Lesser et al. (2024), *Synaptic architecture of leg and wing premotor control networks in Drosophila*: https://doi.org/10.1038/s41586-024-07600-z
- MaleCNS IDs above were checked in this project’s `data/prepared/53e1d528e43cb834982e9207457da777d4c0d55f570844bb96b049ad043c3f7d/neuron-features.parquet`. Cross-dataset IDs are not interchangeable.

All generated circuit assets, observations and checkpoints remain in this project. Original downloaded references remain in shared storage.


## Controlled implementation status

The LTM comparison retains the baseline sensory operator, CB/DN/premotor populations, six arm pools, and their gains. Only the selected gripper motor endpoints and their readout differ. Shared learned arm-edge weights can transfer only after exact topology/sign and input checks. Opening uses below-reference LTM rate deviation in the engineering model; it is not a claim of negative biological firing.

Circuit and calibration evidence: `reports/temporal-control-20260911/ltm-circuit.json` and `ltm-calibration-summary.json`. The static, temporal-memory, and LTM candidates use the same demonstration splits, targets, retinal calibration and optimization budget. See `comparison-plan.json`.

Full results and camera replays: [Temporal and LTM comparison](reports/temporal-control-20260911/README.md).
