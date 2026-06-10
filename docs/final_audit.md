# Final Audit

## Main thesis

Sparse-view Best-of-N inference can amplify geometry errors in learned 3D scene-world models because candidate selection rewards observed-view agreement while ignoring hidden 3D consistency.

## Genuine novelty

The novelty is not sparse-view ambiguity by itself and not generic proxy overoptimization. The narrower contribution is an N-dependent 3D diagnostic: as the candidate count grows, sparse-view proxy selection increasingly chooses view-equivalent hidden-geometry impostors. The repo pairs that diagnostic with a simple coverage/uncertainty-aware reranker.

## Literature coverage

`docs/related_work_matrix.csv` contains 139 linked entries across neural rendering, Gaussian splatting, occupancy and point-cloud representations, 3D scene completion, autonomous-driving world models, multi-hypothesis prediction, proxy overoptimization, uncertainty, and classical multiview geometry. `docs/literature_map.md` records the landscape sweep, serious skim set, and deep-read targets. `docs/hostile_prior_work.md` lists the ten strongest novelty threats.

## Proof status

The paper has a construction-level proposition for sparse-view equivalence and a simple probability argument for N-dependent amplification under independent sampling. It is not a distribution-free theorem for real neural renderers.

## Strongest empirical result

In `results/full/summary.csv`, naive Best-of-N raises proxy score from 0.791 at N=1 to 1.000 at N=64 while true 3D IoU falls from 0.520 to 0.340.

## Strongest diagnostic result

The geometric exploitation gap for naive selection grows from 0.270 at N=1 to 0.660 at N=64. View-impostor selection rises from 20.8% to 100.0%.

## Strongest repair result

At N=16, the repaired reranker achieves true IoU 0.888 versus 0.350 for naive Best-of-N, and reduces hidden error from 0.915 to 0.121.

## Biggest weaknesses

- Synthetic voxel benchmark only.
- Candidate mixture is simulated rather than produced by a trained NeRF, 3DGS, or occupancy world model.
- Repair weights are heuristic.
- The proof assumes independent samples and a simplified projection proxy.
- Related-work coverage is broad but still needs human verification before real submission.

## Paper-readiness judgment

Runnable and coherent as a first autonomous ICLR-style artifact. Not yet ready as a competitive ICLR submission without real neural-scene experiments and stronger ablations.

## Exact PDF path

`C:\Users\wangz\Downloads\best-of-n-3d-scene-world-models.pdf`

## GitHub repo URL

https://github.com/Jason-Wang313/best-of-n-3d-scene-world-models
