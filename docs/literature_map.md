# Literature Map

This map records the literature sweep used to choose the paper angle. The linked CSV contains 139 entries with title, year, venue or arXiv source, link, cluster, contribution, relevance score, threat level, and a short subsumption judgment.

## Sweep Structure

The 100-paper landscape sweep was organized into six clusters:

1. Neural scene representations: NeRF, dynamic NeRFs, sparse-view NeRFs, implicit surfaces, neural volumes, and semantic neural fields.
2. Gaussian and explicit scene representations: 3D Gaussian Splatting, dynamic 3DGS, Gaussian SLAM, geometry-aware splatting, and generative Gaussian scenes.
3. Occupancy, point cloud, and scene-completion world models: occupancy networks, convolutional occupancy networks, point-cloud transformers, semantic scene completion, and autonomous-driving occupancy world models.
4. Generative and latent world models: PlaNet, Dreamer, MuZero, TD-MPC, Genie, driving video world models, and neural simulators.
5. Candidate-pool, multi-hypothesis, decoding, and proxy-selection literature: multiple choice learning, best-of-many objectives, beam search, verifier-guided selection, reward overoptimization, and CEM-style selection.
6. Multiview geometry and uncertainty: visual hulls, space carving, differentiable rendering, calibration, and ensemble uncertainty.

## 30-Paper Serious Skim Set

The serious skim set focused on papers that most constrain the contribution:

- NeRF, pixelNeRF, IBRNet, RegNeRF, DietNeRF, SparseNeRF, MVSNeRF.
- 3D Gaussian Splatting, 2DGS, SuGaR, Deformable 3D Gaussians, Generative Gaussian Splatting.
- Occupancy Networks, Convolutional Occupancy Networks, MonoScene, VoxFormer, OccFormer, OccWorld, SSCNet.
- Scene Representation Networks, Neural Volumes, Neural Scene Flow Fields, Neural Scene Graphs, EmerNeRF.
- Best-of-Many Samples, Multiple Choice Learning, Reward Model Ensembles Help Mitigate Overoptimization, Deep Ensembles.
- A Theory of Shape by Space Carving, Visual Hulls of Curved Objects, Differentiable Volumetric Rendering.

## 20-25-Paper Deep Read Targets

The closest deep-read targets are the ones a reviewer would cite to attack novelty:

- Sparse-view reconstruction and regularization: RegNeRF, SparseNeRF, MVSNeRF, DietNeRF, MonoScene, VoxFormer.
- Modern scene-world models: OccWorld, DriveWorld, GAIA-1, DriveDreamer, VISTA, Generative Gaussian Splatting.
- Geometry ambiguity foundations: visual hulls, space carving, differentiable volumetric rendering.
- Selection and proxy overoptimization: Best-of-Many Samples, Multiple Choice Learning, Reward Model Ensembles, verifier-guided candidate selection, CEM/PETS.
- Uncertainty repair ingredients: Deep Ensembles, calibration, epistemic/aleatoric uncertainty.

## What the Sweep Changed

The initial broad topic could have gone toward a new 3D representation, a new world-model architecture, or a theorem about generic candidate selection. The literature made those options weak:

- New 3D representations are crowded and require real rendering benchmarks.
- Sparse-view regularizers are well studied; claiming a generic fix would be easy to dismiss.
- Classical visual hull literature already proves sparse-view ambiguity, so the novelty cannot be "sparse views are ambiguous."
- Reward overoptimization literature already says optimizing a proxy can fail, so the novelty cannot be "candidate selection overfits a proxy."

The defensible center is narrower: **in learned 3D scene-world candidate selection, candidate-pool pressure can turn sparse-view ambiguity into a view-impostor failure mode.** The contribution is a mechanism-specific diagnostic and a small repair, not a universal method.

## Chosen Angle

The chosen paper shape is:

mechanism -> diagnostic -> synthetic empirical validation -> uncertainty/coverage-aware repair.

The final claim is architecture-specific but deliberately modest:

> When a 3D scene-world model proposes multiple candidate occupancy/neural-scene futures and a sparse view proxy selects the best candidate, increasing N can monotonically improve the proxy while worsening global 3D consistency, hidden-region IoU, and candidate diversity. A reranker using ensemble consensus and view-coverage plausibility can reduce the exploitation gap in this benchmark.

## Open Literature Gaps

- The repo does not yet run on real NeRF, 3DGS, or autonomous-driving occupancy datasets.
- The repair is a reranker, not a trained model or representation.
- The theoretical proposition is diagnostic and construction-based; it is not a distribution-free guarantee for real systems.
- The literature sweep is broad and linked, but this first pass still needs a human second pass over exact experimental protocols before submission.
