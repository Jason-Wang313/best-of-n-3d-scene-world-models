from __future__ import annotations

import numpy as np
import pandas as pd

from view_impostor_audit.experiment import ExperimentConfig, run_experiment
from view_impostor_audit.geometry import iou, project_silhouette, visible_mask, visual_hull_from_scene
from view_impostor_audit.scene import generate_scene, sample_candidates
from view_impostor_audit.scoring import proxy_score, score_candidates, select_index


def test_scene_generation_is_reproducible() -> None:
    seed = 123
    a = generate_scene(np.random.default_rng(seed), grid=18, seed=seed)
    b = generate_scene(np.random.default_rng(seed), grid=18, seed=seed)
    assert np.array_equal(a.occupancy, b.occupancy)
    assert 0.02 < a.occupancy.mean() < 0.35


def test_projection_and_visible_mask_shapes() -> None:
    rng = np.random.default_rng(4)
    scene = generate_scene(rng, grid=16)
    assert project_silhouette(scene.occupancy, "x").shape == (16, 16)
    mask = visible_mask(scene.occupancy, ("x", "y"))
    hull = visual_hull_from_scene(scene.occupancy, ("x", "y"))
    assert mask.shape == scene.occupancy.shape
    assert np.all(mask <= scene.occupancy)
    assert hull.sum() >= scene.occupancy.sum()


def test_view_impostors_score_well_under_sparse_proxy() -> None:
    rng = np.random.default_rng(99)
    scene = generate_scene(rng, grid=18)
    candidates = sample_candidates(scene.occupancy, 80, np.random.default_rng(101), proxy_axes=("x", "y"))
    impostors = [c for c in candidates if c.mode == "view_impostor"]
    assert impostors
    best_impostor = max(impostors, key=lambda c: proxy_score(c.occupancy, scene.occupancy, ("x", "y")))
    assert proxy_score(best_impostor.occupancy, scene.occupancy, ("x", "y")) > 0.96
    assert iou(best_impostor.occupancy, scene.occupancy) < 0.55


def test_repair_can_avoid_sparse_view_impostor() -> None:
    rng = np.random.default_rng(314)
    scene = generate_scene(rng, grid=20)
    candidates = sample_candidates(scene.occupancy, 64, np.random.default_rng(315), proxy_axes=("x", "y"))
    scores = score_candidates(candidates, scene.occupancy, ("x", "y"))
    naive = candidates[select_index(scores, "naive")]
    repaired = candidates[select_index(scores, "repaired")]
    assert proxy_score(naive.occupancy, scene.occupancy, ("x", "y")) >= proxy_score(
        repaired.occupancy, scene.occupancy, ("x", "y")
    ) - 1e-9
    assert iou(repaired.occupancy, scene.occupancy) >= iou(naive.occupancy, scene.occupancy)


def test_smoke_experiment_generates_expected_files(tmp_path) -> None:
    paths = run_experiment(
        ExperimentConfig(seed=7, grid=16, scenes=5, n_values=(1, 2, 4, 8), proxy_axes=("x", "y")),
        tmp_path,
        figures_dir=tmp_path / "figures",
    )
    summary = pd.read_csv(paths["summary"])
    assert {"method", "n", "proxy_score_mean", "true_iou_mean", "exploitation_gap_mean"} <= set(summary.columns)
    assert paths["tradeoff"].exists()
    naive = summary[summary.method == "naive"].sort_values("n")
    assert naive.proxy_score_mean.iloc[-1] >= naive.proxy_score_mean.iloc[0]
