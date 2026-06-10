"""Experiment orchestration for the Best-of-N 3D scene benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import Iterable

import numpy as np
import pandas as pd

from bon3d.metrics import evaluate_selection
from bon3d.plotting import plot_example, plot_hidden_and_diversity, plot_tradeoff
from bon3d.scene import generate_scene, sample_candidates
from bon3d.scoring import score_candidates, select_index


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 1729
    grid: int = 24
    scenes: int = 64
    n_values: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    proxy_axes: tuple[str, ...] = ("x", "y")


PRESETS = {
    "smoke": ExperimentConfig(seed=1729, grid=20, scenes=10, n_values=(1, 2, 4, 8, 16)),
    "full": ExperimentConfig(seed=1729, grid=24, scenes=72, n_values=(1, 2, 4, 8, 16, 32, 64)),
}


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    sem = lambda x: float(x.std(ddof=1) / np.sqrt(max(len(x), 1)))  # noqa: E731
    grouped = metrics.groupby(["method", "n"], as_index=False).agg(
        proxy_score_mean=("proxy_score", "mean"),
        proxy_score_sem=("proxy_score", sem),
        true_iou_mean=("true_iou", "mean"),
        true_iou_sem=("true_iou", sem),
        hidden_error_mean=("hidden_error", "mean"),
        hidden_error_sem=("hidden_error", sem),
        exploitation_gap_mean=("exploitation_gap", "mean"),
        exploitation_gap_sem=("exploitation_gap", sem),
        top_proxy_diversity_mean=("top_proxy_diversity", "mean"),
        visible_occupancy_fraction_mean=("visible_occupancy_fraction", "mean"),
        view_impostor_rate=("selected_mode", lambda s: float((s == "view_impostor").mean())),
    )
    return grouped.sort_values(["method", "n"]).reset_index(drop=True)


def _candidate_trace(scene_id: int, n_value: int, candidates, scores) -> list[dict[str, float | int | str]]:
    rows = []
    for candidate, score in zip(candidates, scores):
        rows.append(
            {
                "scene_id": scene_id,
                "n": n_value,
                "candidate_index": candidate.index,
                "mode": candidate.mode,
                "proxy_score": score.proxy,
                "repair_score": score.repair,
                "consensus_score": score.consensus,
                "volume_plausibility": score.volume_plausibility,
                "surface_penalty": score.surface_penalty,
                "occupancy_fraction": float(candidate.occupancy.mean()),
            }
        )
    return rows


def run_experiment(
    config: ExperimentConfig,
    output: str | Path,
    figures_dir: str | Path | None = "figures",
) -> dict[str, Path]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figures_path = Path(figures_dir) if figures_dir is not None else output / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    max_n = max(config.n_values)
    rows: list[dict[str, float | int | str]] = []
    trace_rows: list[dict[str, float | int | str]] = []
    example_payload = None

    for scene_id in range(config.scenes):
        scene_rng = np.random.default_rng(config.seed + 1009 * scene_id)
        cand_rng = np.random.default_rng(config.seed + 7919 * scene_id + 17)
        scene = generate_scene(scene_rng, grid=config.grid, seed=config.seed + scene_id)
        all_candidates = sample_candidates(scene.occupancy, max_n, cand_rng, proxy_axes=config.proxy_axes)

        for n_value in config.n_values:
            subset = all_candidates[:n_value]
            scores = score_candidates(subset, scene.occupancy, config.proxy_axes)
            naive_index = select_index(scores, "naive")
            repaired_index = select_index(scores, "repaired")
            rows.append(
                evaluate_selection(
                    scene.occupancy, subset, scores, naive_index, config.proxy_axes, "naive", n_value, scene_id
                )
            )
            rows.append(
                evaluate_selection(
                    scene.occupancy,
                    subset,
                    scores,
                    repaired_index,
                    config.proxy_axes,
                    "repaired",
                    n_value,
                    scene_id,
                )
            )
            if n_value == max_n:
                trace_rows.extend(_candidate_trace(scene_id, n_value, subset, scores))
                if example_payload is None and subset[naive_index].mode == "view_impostor":
                    example_payload = (scene.occupancy, subset[naive_index].occupancy, subset[repaired_index].occupancy)

    metrics = pd.DataFrame(rows)
    summary = summarize(metrics)
    trace = pd.DataFrame(trace_rows)

    metrics_path = output / "metrics.csv"
    summary_path = output / "summary.csv"
    trace_path = output / "candidate_trace.csv"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    trace.to_csv(trace_path, index=False)

    tradeoff_path = figures_path / "bon_tradeoff.png"
    hidden_path = figures_path / "hidden_diversity.png"
    plot_tradeoff(summary, tradeoff_path)
    plot_hidden_and_diversity(summary, hidden_path)
    if example_payload is not None:
        example_path = figures_path / "example_projections.png"
        plot_example(*example_payload, output=example_path)
    else:
        example_path = figures_path / "example_projections.png"

    # Keep a copy beside the output so each run is self-contained.
    for figure in [tradeoff_path, hidden_path, example_path]:
        if figure.exists():
            copyfile(figure, output / figure.name)

    return {
        "metrics": metrics_path,
        "summary": summary_path,
        "trace": trace_path,
        "tradeoff": tradeoff_path,
        "hidden_diversity": hidden_path,
        "example": example_path,
    }
