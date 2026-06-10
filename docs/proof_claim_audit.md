# Proof and Claim Audit

## Formal Claim A: Sparse-view equivalence

**Claim:** For a fixed finite set of projection views, there can exist two distinct 3D occupancies with identical sparse proxy score and different true 3D IoU.

**Status:** Supported by construction and classical visual-hull theory.

**Reasoning:** A sparse silhouette/front-depth proxy constrains only the first visible occupied voxel and whether each ray hits occupancy. Hidden voxels behind the first hit can be changed without changing the proxy, as long as no new front surface appears. Therefore the proxy is not injective over 3D occupancy.

**Weakness:** The construction uses orthographic axis views and binary occupancy. Extension to photometric NeRF/3DGS scoring requires assumptions about opacity, rendering tolerance, and view discretization.

## Formal Claim B: N-dependent selection amplification

**Claim:** If a candidate sampler has nonzero probability p of producing a proxy-perfect but hidden-inconsistent candidate, then the probability that naive Best-of-N sees at least one such candidate is 1 - (1 - p)^N, increasing in N.

**Status:** Proven under independent sampling and fixed p.

**Proof sketch:** The event of no impostor in N independent samples has probability (1 - p)^N. Its complement has probability 1 - (1 - p)^N. If the impostor's proxy score exceeds the faithful candidate scores, the sparse proxy selector chooses an impostor whenever one appears.

**Weakness:** Real world-model samples may be correlated, and proxy-perfect impostors may not always dominate. The benchmark does not require perfection; the empirical curves show approximate dominance.

## Empirical Claim C: Naive Best-of-N worsens true 3D consistency in the benchmark

**Claim:** In the full run, increasing N improves sparse proxy score but worsens true 3D IoU and hidden-region error for naive selection.

**Status:** Supported by `results/full/summary.csv`.

**Numbers:** Naive proxy score rises from 0.791 at N=1 to 1.000 at N=64. True IoU falls from 0.520 to 0.340. Hidden error rises from 0.442 to 0.929. View-impostor selection rises from 0.208 to 1.000.

**Weakness:** Synthetic distributions are hand-designed. Need real neural-scene validation.

## Empirical Claim D: Repair reduces the exploitation gap

**Claim:** The coverage-aware reranker reduces exploitation gap and hidden error relative to naive Best-of-N.

**Status:** Supported in the benchmark.

**Numbers:** At N=16, naive true IoU is 0.350 and exploitation gap is 0.650. Repaired true IoU is 0.888 and exploitation gap is 0.074. Hidden error drops from 0.915 to 0.121.

**Weakness:** Repair weights are fixed heuristics and may fail under adversarial candidate sets whose hidden geometry matches the volume/surface priors.

## Claims Excluded

- No claim that all Best-of-N scene-world models fail.
- No claim that the repair is optimal.
- No claim that the synthetic benchmark predicts real-world magnitude.
- No claim that sparse-view regularization papers are insufficient for reconstruction; they address a different training-time problem.
