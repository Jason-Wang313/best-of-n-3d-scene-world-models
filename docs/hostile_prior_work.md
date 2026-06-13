# Hostile Prior Work

These are the ten papers or paper families reviewers are most likely to use to argue that the project is not new.

## 1. Visual hull and space carving

**Papers:** Visual Hulls of Curved Objects; A Theory of Shape by Space Carving; Space Carving: Theory and Practice.

**Attack:** Classical multiview geometry already says sparse silhouettes and projections underdetermine 3D shape.

**Response:** Correct. The paper does not claim that ambiguity is new. The new mechanism is that a learned multi-candidate scene-world model plus sparse-view proxy selection converts this ambiguity into a candidate-pool failure mode: rare sparse-view impostors become more likely and are selected more often as N grows.

## 2. RegNeRF, SparseNeRF, DietNeRF, MVSNeRF

**Attack:** Sparse-view NeRF papers already study overfitting and geometry regularization.

**Response:** These methods regularize reconstruction or improve generalizable rendering. They do not study inference-time selection among sampled future scene candidates, and they do not report proxy improvement paired with true 3D degradation as N increases.

## 3. MonoScene, SSCNet, VoxFormer, OccFormer

**Attack:** Scene completion already studies hidden geometry from sparse observations.

**Response:** Scene completion predicts a scene; this work studies selection over many predicted scenes under a sparse scorer. The failure is not merely hidden-region error, but the amplification of hidden-region error by candidate-pool pressure.

## 4. OccWorld and DriveWorld

**Attack:** 3D occupancy world models already exist.

**Response:** These papers motivate the architecture class. They do not isolate sparse-view candidate selection or a proxy reranking hazard. This repo can be viewed as a diagnostic benchmark for such models.

## 5. Generative Gaussian Splatting and dynamic 3DGS

**Attack:** Generative 3DGS already produces 3D-consistent candidate scenes.

**Response:** The paper is representation-agnostic: a Gaussian candidate can also be scored through sparse views. The benchmark uses voxels for controlled measurement, but the mechanism applies to any representation where many 3D states share sparse projections.

## 6. Best-of-Many Samples and Multiple Choice Learning

**Attack:** Multi-hypothesis prediction and best-of-many objectives are already known.

**Response:** Those works train or evaluate multiple outputs. They do not identify a geometric sparse-view exploit where the best proxy candidate becomes less globally consistent as N increases.

## 7. Reward overoptimization and verifier-guided candidate selection

**Attack:** Proxy overoptimization under candidate selection is already known from reward models and verifiers.

**Response:** The project borrows that conceptual lens. The novelty is the 3D mechanism: sparse views create equivalence classes of hidden geometry, and candidate-pool pressure preferentially selects members that overfit the observed projections.

## 8. Deep ensembles and uncertainty repair

**Attack:** Ensemble uncertainty and consensus are standard.

**Response:** The repair is intentionally simple and not claimed as a new uncertainty estimator. Its contribution is to test whether uncertainty/coverage signals target the diagnosed failure mode without hidden ground truth.

## 9. Differentiable rendering and projective consistency losses

**Attack:** Projective consistency has long been used to train 3D representations.

**Response:** Projective consistency can be part of the problem when it is the proxy. The paper distinguishes sparse observed-view consistency from global 3D consistency, and measures the gap under selection.

## 10. Real 3D benchmarks and neural renderers

**Attack:** The empirical evidence is synthetic.

**Response:** This is the strongest limitation. The current artifact is a mechanism paper and reproducible diagnostic, not a full benchmark claim about real NeRF/3DGS deployments. The final audit marks real-system validation as the main missing step.
