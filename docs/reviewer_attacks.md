# Reviewer Attacks

## Attack 1: This is just Goodhart's law.

**Answer:** It is a 3D-specific Goodhart mechanism, not just a slogan. The diagnostic relies on sparse-view equivalence classes: candidates can be identical under the proxy views and different in hidden occupancy. The experiments measure the N-dependent shift from faithful candidates to view impostors.

## Attack 2: The benchmark cheats by making impostors.

**Answer:** The benchmark intentionally includes a failure mode because it is a mechanism study. The key question is whether sparse-view scoring selects that mode more often as N grows. Real generative scene models can also produce candidates with correct visible surfaces and uncertain hidden geometry, but this repo does not claim the exact mixture rates transfer.

## Attack 3: The repair uses the target.

**Answer:** The repair uses the same sparse silhouettes/depths available to the proxy, plus candidate ensemble consensus and a visual-hull volume prior. It does not access hidden target occupancy. The evaluation metrics use hidden target occupancy only after selection.

## Attack 4: The synthetic scene generator is too simple.

**Answer:** True as a limitation of the main replicated run. The v4 artifact adds a CPU-light ModelNet10-derived chair/monitor voxel tier, so the mechanism is no longer only procedural primitives. Full trained NeRF/3DGS or autonomous-driving occupancy validation remains outside scope.

## Attack 5: The proxy is unrealistic.

**Answer:** The proxy is a simplified stand-in for sparse render, view, or partial-geometry scoring. It combines silhouettes and first-hit depth, which are common projective signals. The exact proxy should be replaced with task-specific render losses in real experiments.

## Attack 6: The repair is hand-tuned.

**Answer:** The repair weights are fixed and simple; no hidden labels tune them. Still, it is a proof-of-concept rather than a final method. A stronger version would learn calibration from held-out scenes and report ablations.

## Attack 7: Why not just score more views?

**Answer:** More views are indeed a solution when available. The paper addresses settings where only sparse scoring views or partial observations are available at inference time. The repair approximates this by asking whether candidate consensus and coverage plausibility help without hidden ground truth.

## Attack 8: True IoU is not available in real deployments.

**Answer:** Correct. True IoU is an evaluation diagnostic, not a deployment signal. The deployable part is the reranker; the benchmark uses true IoU to show why the proxy is unsafe.

## Attack 9: Candidate-pool selection should improve if the model is calibrated.

**Answer:** If proxy score equals true utility and the candidate distribution is calibrated, larger pools can help. The paper studies the common failure case where the proxy omits hidden geometry and the candidate distribution contains view-equivalent impostors.

## Attack 10: This is not enough for ICLR.

**Answer:** The v4 artifact is stronger than a synthetic-only pass: it includes a replicated synthetic benchmark, compact high-pool stress suite, ModelNet10 real-shape tier, ablations, calibration diagnostics, and a failing claim audit for unsupported edits. Larger trained neural-scene experiments and learned uncertainty calibration remain the next scope increase.
