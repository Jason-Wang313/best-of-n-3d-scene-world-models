# Novelty Decision

## Candidate Contributions Considered

1. **A new 3D scene representation.** Rejected. The literature is dense, and a new representation would require real rendering benchmarks beyond a first autonomous pass.
2. **A theorem about all Best-of-N world models.** Rejected. A broad theorem would either be trivial proxy overoptimization or too assumption-heavy.
3. **A diagnostic for sparse-view Best-of-N geometry exploitation.** Chosen. It is narrow, testable, architecture-specific, and under-covered by prior work.
4. **A new repair method.** Secondary. The repair is useful but not strong enough to be the main novelty.

## Final Thesis

Sparse-view Best-of-N inference in learned 3D scene-world models can amplify geometry errors. As the candidate count increases, a proxy scorer that only checks a few views is increasingly likely to select a candidate that matches those views while disagreeing with the target in hidden 3D regions.

## Mechanism

For any fixed sparse set of views, many distinct 3D occupancies share the same silhouettes and first visible surfaces. If a stochastic world model sometimes samples such view-equivalent but hidden-inconsistent candidates, the probability that at least one appears grows with N. A sparse proxy then selects it because its observed-view score is high, even though true 3D IoU is low.

## Why This Is Not Subsumed

- Classical geometry supplies the ambiguity, but not the N-dependent learned-selection behavior.
- Sparse-view NeRF and occupancy papers repair reconstruction, but do not study Best-of-N selection over sampled future scenes.
- Reward overoptimization papers study proxy failures, but not the view-equivalence structure of 3D scene representations.
- Multi-hypothesis papers study diversity and best-of-many losses, but not sparse geometry impostors.

## Chosen Contribution Type

The strongest contribution type is **diagnostic plus repair**:

- Diagnostic: geometric exploitation gap = sparse proxy score minus true 3D IoU.
- Empirical phenomenon: in the full synthetic run, naive proxy score rises from 0.791 at N=1 to 1.000 at N=64 while true 3D IoU falls from 0.520 to 0.340.
- Repair: ensemble-consensus and view-coverage-aware reranking raises true IoU to 0.888 at N=16 and keeps the exploitation gap far below naive Best-of-N.

## Claim Boundary

The paper may claim:

- The mechanism exists in a controlled 3D scene-world benchmark.
- The diagnostic detects proxy-view exploitation.
- The repair reduces the exploitation gap in this benchmark.

The paper may not yet claim:

- Real NeRF, 3DGS, or autonomous-driving models always exhibit this failure.
- The repair is optimal or universally reliable.
- The result is a complete ICLR-ready empirical story without real-data validation.
