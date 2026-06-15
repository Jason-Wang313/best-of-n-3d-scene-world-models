# Final Audit

## Main thesis

Sparse-view candidate selection can amplify hidden-geometry errors in learned
3D scene-world models because projective proxy scoring rewards observed-view
agreement while ignoring underdetermined occupancy behind first-hit surfaces.

## v4 contribution

This is no longer framed as a generic candidate-selection paper. The v4
manuscript is a geometry-specific view-impostor audit with visual-hull masks,
hidden-region metrics, pool-size curves, repair ablations, calibration
diagnostics, failure slices, a real-shape ModelNet10 benchmark tier, and a
real-system porting protocol.

## Evidence split

- `results/full/summary.csv` is the replicated 72-scene statistical backbone.
- `results/expansion/` is a compact high-pool stress suite with pool size 256,
  view-axis stress, sampler-mixture stress, hidden-mass stress, ablations,
  candidate calibration, and failure cases.
- `results/expansion/claims.json` passes after combining the replicated full
  benchmark with the compact stress suite.
- `results/modelnet10_benchmark/claims.json` passes on a tiny ModelNet10-derived
  chair/monitor voxel slice.

## Strongest replicated result

In `results/full/summary.csv`, the sparse-view selector raises proxy score from
0.791 at N=1 to 1.000 at N=64 while true 3D IoU falls from 0.520 to 0.340 and
hidden error rises from 0.442 to 0.929.

## Strongest repair result

At N=64 in the replicated run, the reranked selector reaches true IoU 0.851
versus 0.340 for sparse-view selection and reduces the exploitation gap from
0.660 to 0.127.

## High-pool stress result

The compact N=256 stress run confirms proxy saturation and shows that the
repair improves true IoU and gap, but also exposes remaining repaired failures.
This stress suite is reported as diagnostic evidence, not as a replacement for
the 72-scene replicated benchmark.

## Real-shape benchmark result

On the ModelNet10-derived chair/monitor slice, raw sparse-view selection raises
proxy score by 0.167 from N=1 to N=64 while true 3D IoU drops by 0.130 and
hidden-region IoU drops by 0.458. The coverage-aware reranker recovers 0.420
true-IoU points over raw N=64 selection and reduces the exploitation gap by
0.455.

## Biggest weaknesses

- Synthetic voxel benchmark remains the main replicated evidence.
- Candidate mixture is simulated rather than produced by a trained NeRF, 3DGS,
  or occupancy world model.
- Repair weights are heuristic.
- The compact N=256 stress suite is small because high-pool replication was too
  slow for the available CPU budget.
- Full trained NeRF, 3DGS, and autonomous-driving occupancy validation remains
  outside this scoped artifact.

## Paper-readiness judgment

Submission-ready v4 as a scoped mechanism and audit paper. The paper is 25+
pages, explicitly avoids duplicate-wrapper framing, and keeps claims bounded to
the replicated synthetic, compact stress, and ModelNet10 real-shape evidence.

## Exact PDF paths

Repository final:
`C:\Users\wangz\best-of-n-3d-scene-world-models\paper\final\best-of-n-3d-scene-world-models-v4.pdf`

Desktop final:
`C:\Users\wangz\OneDrive\Desktop\best-of-n-3d-scene-world-models-v4.pdf`

## GitHub repo URL

https://github.com/Jason-Wang313/best-of-n-3d-scene-world-models
