# View-Impostor Audits for 3D Scene World Models

This repository contains an anonymous ICLR-style research artifact studying
view-impostor failures in learned 3D / neural scene-representation world
models.

**Thesis.** In sparse-view 3D scene prediction, larger candidate pools can
amplify proxy-view errors: as `N` grows, a scorer that sees only a few projected
views increasingly selects candidates that match those views while becoming less
globally consistent in 3D. The repository demonstrates the mechanism on a
deterministic synthetic voxel benchmark and tests a coverage/uncertainty-aware
reranker.

The code is intentionally CPU-friendly. It uses primitive voxel scenes, a
simulated scene-world predictor, sparse orthographic proxy views, and explicit
diagnostics for proxy fit, true 3D consistency, hidden-region error, diversity
collapse, and geometric exploitation gap.

## Quick Start

```bash
python -m pytest
python experiments/run_synthetic.py --preset smoke --output results/smoke
python experiments/run_synthetic.py --preset full --output results/full
python experiments/run_expansion_suite.py --mode full --output results/expansion
python scripts/build_paper.py
python scripts/run_claim_audit.py
```

The paper build script writes:

- `paper/final/best-of-n-3d-scene-world-models-v3.pdf`
- `C:\Users\wangz\OneDrive\Desktop\best-of-n-3d-scene-world-models-v3.pdf`

## Repository Layout

- `src/view_impostor_audit/`: benchmark, scoring, metrics, and plotting code.
- `experiments/run_synthetic.py`: CLI for smoke and full experiments.
- `tests/`: unit and smoke tests.
- `docs/`: literature map, novelty decision, reviewer attacks, proof audit, and final audit.
- `paper/`: anonymous LaTeX submission.
- `results/`, `figures/`: generated tables and figures.

## What the Synthetic Benchmark Models

The benchmark is not a claim that real NeRF or Gaussian-splatting systems fail
in exactly this toy way. It isolates a mechanism common to sparse-view scoring:
many 3D volumes can share the same front-depth and silhouette projections.
When a learned predictor samples multiple candidate scene futures, rare
view-impostor candidates become more likely as `N` increases. A sparse proxy
scorer can then prefer the impostor even when its hidden geometry is poor.

The repair is deliberately modest: it keeps the same candidate set but reranks
with proxy fit, ensemble geometry consensus, visual-hull volume plausibility,
and surface-complexity penalties. This tests whether a cheap uncertainty and
coverage signal can reduce proxy exploitation without needing ground-truth
hidden geometry at test time.

## v3 Submission Artifact

The v3 manuscript is a 25+ page submission-ready paper. The replicated
72-scene benchmark in `results/full` is the main statistical evidence. The
compact v3 stress suite in `results/expansion` adds pool-size-256 diagnostics,
view-axis stress, sampler-mixture stress, hidden-mass stress, repair ablations,
candidate-level calibration, failure slices, and a machine-readable
`claims.json` audit.
