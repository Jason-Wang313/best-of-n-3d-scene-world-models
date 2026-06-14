from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from view_impostor_audit.geometry import iou
from view_impostor_audit.metrics import evaluate_selection
from view_impostor_audit.scene import SamplerConfig, generate_scene, sample_candidates
from view_impostor_audit.scoring import CandidateScore, proxy_score, score_candidates


@dataclass(frozen=True)
class SuiteConfig:
    mode: str
    main_grid: int
    main_scenes: int
    main_seeds: tuple[int, ...]
    main_n_values: tuple[int, ...]
    stress_grid: int
    stress_scenes: int
    stress_seeds: tuple[int, ...]
    stress_n_values: tuple[int, ...]


PRESETS = {
    "quick": SuiteConfig(
        mode="quick-smoke",
        main_grid=14,
        main_scenes=5,
        main_seeds=(1729,),
        main_n_values=(1, 4, 8),
        stress_grid=14,
        stress_scenes=4,
        stress_seeds=(1729,),
        stress_n_values=(4, 8),
    ),
    "full": SuiteConfig(
        mode="full-v3",
        main_grid=20,
        main_scenes=4,
        main_seeds=(1729,),
        main_n_values=(1, 16, 64, 256),
        stress_grid=18,
        stress_scenes=1,
        stress_seeds=(1729,),
        stress_n_values=(256,),
    ),
}


BASE_AXES = ("x", "y")
ALL_AXES = ("x", "y", "z")


def _axis_label(axes: Iterable[str]) -> str:
    return "".join(tuple(axes))


def _sem(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def _summarize(rows: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    summary = rows.groupby(list(group_cols), as_index=False, dropna=False).agg(
        proxy_score_mean=("proxy_score", "mean"),
        proxy_score_sem=("proxy_score", _sem),
        true_iou_mean=("true_iou", "mean"),
        true_iou_sem=("true_iou", _sem),
        hidden_error_mean=("hidden_error", "mean"),
        hidden_error_sem=("hidden_error", _sem),
        exploitation_gap_mean=("exploitation_gap", "mean"),
        exploitation_gap_sem=("exploitation_gap", _sem),
        surface_fraction_mean=("surface_fraction", "mean"),
        visible_occupancy_fraction_mean=("visible_occupancy_fraction", "mean"),
        view_impostor_rate=("selected_mode", lambda s: float((s == "view_impostor").mean())),
        scenes=("scene_id", "count"),
    )
    return summary.sort_values(list(group_cols)).reset_index(drop=True)


def _custom_value(score: CandidateScore, method: str) -> float:
    if method == "naive":
        return score.proxy
    if method == "repaired":
        return score.repair
    if method == "no_consensus":
        return 0.70 * score.proxy + 0.30 * score.volume_plausibility - 0.12 * score.surface_penalty
    if method == "no_volume":
        return 0.65 * score.proxy + 0.27 * score.consensus - 0.12 * score.surface_penalty
    if method == "no_surface_penalty":
        return 0.58 * score.proxy + 0.27 * score.consensus + 0.22 * score.volume_plausibility
    if method == "consensus_only":
        return score.consensus
    if method == "volume_surface_only":
        return 0.75 * score.volume_plausibility - 0.25 * score.surface_penalty
    raise ValueError(f"unknown custom method: {method}")


def _select_index(
    method: str,
    candidates,
    scores: Sequence[CandidateScore],
    target: np.ndarray,
    axes: tuple[str, ...],
    rng: np.random.Generator,
) -> int:
    if method == "random":
        return int(rng.integers(0, len(candidates)))
    if method == "extra_view_proxy":
        values = [proxy_score(candidate.occupancy, target, ALL_AXES) for candidate in candidates]
        return int(np.argmax(values))
    values = [_custom_value(score, method) for score in scores]
    return int(np.argmax(values))


def _evaluate_methods(
    *,
    target: np.ndarray,
    candidates,
    scores: Sequence[CandidateScore],
    axes: tuple[str, ...],
    n_value: int,
    scene_id: int,
    seed: int,
    regime: str,
    sampler_label: str,
    methods: Sequence[str],
    rng: np.random.Generator,
    extra: dict[str, float | int | str] | None = None,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for method in methods:
        selected = _select_index(method, candidates, scores, target, axes, rng)
        row = evaluate_selection(target, candidates, scores, selected, axes, method, n_value, scene_id)
        row.update(
            {
                "seed": seed,
                "regime": regime,
                "axis_label": _axis_label(axes),
                "sampler_label": sampler_label,
            }
        )
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def _run_regime(
    *,
    regime: str,
    grid: int,
    scenes: int,
    seeds: Sequence[int],
    n_values: Sequence[int],
    axes: tuple[str, ...],
    sampler: SamplerConfig,
    sampler_label: str,
    methods: Sequence[str],
    collect_candidates: bool = False,
    extra: dict[str, float | int | str] | None = None,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    rows: list[dict[str, float | int | str]] = []
    candidate_rows: list[dict[str, float | int | str]] = []
    max_n = max(n_values)
    for seed in seeds:
        for local_scene_id in range(scenes):
            scene_id = seed * 100000 + local_scene_id
            scene_rng = np.random.default_rng(seed + 1009 * local_scene_id)
            cand_rng = np.random.default_rng(seed + 7919 * local_scene_id + 17)
            selector_rng = np.random.default_rng(seed + 1543 * local_scene_id + 91)
            scene = generate_scene(scene_rng, grid=grid, seed=scene_id)
            candidates = sample_candidates(scene.occupancy, max_n, cand_rng, proxy_axes=axes, sampler=sampler)

            for n_value in n_values:
                subset = candidates[:n_value]
                scores = score_candidates(subset, scene.occupancy, axes)
                rows.extend(
                    _evaluate_methods(
                        target=scene.occupancy,
                        candidates=subset,
                        scores=scores,
                        axes=axes,
                        n_value=n_value,
                        scene_id=scene_id,
                        seed=seed,
                        regime=regime,
                        sampler_label=sampler_label,
                        methods=methods,
                        rng=selector_rng,
                        extra=extra,
                    )
                )

            if collect_candidates:
                scores = score_candidates(candidates, scene.occupancy, axes)
                for candidate, score in zip(candidates, scores):
                    candidate_rows.append(
                        {
                            "seed": seed,
                            "scene_id": scene_id,
                            "regime": regime,
                            "axis_label": _axis_label(axes),
                            "sampler_label": sampler_label,
                            "candidate_index": candidate.index,
                            "candidate_mode": candidate.mode,
                            "proxy_score": score.proxy,
                            "repair_score": score.repair,
                            "consensus_score": score.consensus,
                            "volume_plausibility": score.volume_plausibility,
                            "surface_penalty": score.surface_penalty,
                            "true_iou": iou(candidate.occupancy, scene.occupancy),
                            "n": max_n,
                        }
                    )
    return rows, candidate_rows


def _plot_candidate_count(summary: pd.DataFrame, output: Path) -> None:
    main = summary[(summary.regime == "candidate_count") & (summary.axis_label == "xy")]
    methods = ["naive", "repaired", "extra_view_proxy", "random"]
    colors = {
        "naive": "#b23a48",
        "repaired": "#2f7f6f",
        "extra_view_proxy": "#3f5f9f",
        "random": "#6f6f6f",
    }
    labels = {
        "naive": "sparse-view",
        "repaired": "geometry rerank",
        "extra_view_proxy": "extra-view control",
        "random": "random",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6), constrained_layout=True)
    for method in methods:
        group = main[main.method == method].sort_values("n")
        if group.empty:
            continue
        axes[0].plot(group.n, group.proxy_score_mean, marker="o", color=colors[method], label=labels[method])
        axes[1].plot(group.n, group.true_iou_mean, marker="o", color=colors[method])
        axes[2].plot(group.n, group.exploitation_gap_mean, marker="o", color=colors[method])
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("candidate count N")
    axes[0].set_ylabel("proxy score")
    axes[1].set_ylabel("true 3D IoU")
    axes[2].set_ylabel("proxy - true IoU")
    axes[0].set_title("Sparse score")
    axes[1].set_title("Global occupancy")
    axes[2].set_title("Exploitation gap")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_axis_sweep(summary: pd.DataFrame, output: Path) -> None:
    available = summary[summary.regime == "axis_sweep"]
    max_n = int(available.n.max()) if not available.empty else 0
    data = available[available.n == max_n]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), constrained_layout=True)
    labels = ["x", "xy", "xyz"]
    x = np.arange(len(labels))
    width = 0.35
    for offset, method, color in [(-width / 2, "naive", "#b23a48"), (width / 2, "repaired", "#2f7f6f")]:
        group = data[data.method == method].set_index("axis_label").reindex(labels)
        axes[0].bar(x + offset, group.true_iou_mean, width, color=color, label=method)
        axes[1].bar(x + offset, group.exploitation_gap_mean, width, color=color, label=method)
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlabel("scored view axes")
    axes[0].set_ylabel("true 3D IoU")
    axes[1].set_ylabel("proxy - true IoU")
    axes[0].set_title("Adding observations")
    axes[1].set_title("Remaining gap")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_mixture_stress(summary: pd.DataFrame, output: Path) -> None:
    available = summary[summary.regime == "mixture_stress"]
    max_n = int(available.n.max()) if not available.empty else 0
    data = available[available.n == max_n].copy()
    data["impostor_prior"] = data["impostor_prior"].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.5), constrained_layout=True)
    for method, color in [("naive", "#b23a48"), ("repaired", "#2f7f6f")]:
        group = data[data.method == method].sort_values("impostor_prior")
        axes[0].plot(group.impostor_prior, group.true_iou_mean, marker="o", color=color, label=method)
        axes[1].plot(group.impostor_prior, group.view_impostor_rate, marker="o", color=color, label=method)
    axes[0].set_ylabel("true 3D IoU")
    axes[1].set_ylabel("selected impostor rate")
    for ax in axes:
        ax.set_xlabel("sampler impostor prior")
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Robustness to mixture shift")
    axes[1].set_title("Selector failure rate")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_repair_ablation(ablation: pd.DataFrame, output: Path) -> None:
    max_n = int(ablation.n.max()) if not ablation.empty else 0
    data = ablation[(ablation.n == max_n)].copy()
    order = [
        "naive",
        "repaired",
        "no_consensus",
        "no_volume",
        "no_surface_penalty",
        "consensus_only",
        "volume_surface_only",
        "random",
    ]
    data = data.set_index("method").reindex(order).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), constrained_layout=True)
    colors = ["#b23a48" if m == "naive" else "#2f7f6f" if m == "repaired" else "#667085" for m in data.method]
    axes[0].bar(np.arange(len(data)), data.true_iou_mean, color=colors)
    axes[1].bar(np.arange(len(data)), data.exploitation_gap_mean, color=colors)
    for ax in axes:
        ax.set_xticks(np.arange(len(data)), data.method, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("true 3D IoU")
    axes[1].set_ylabel("proxy - true IoU")
    axes[0].set_title("Repair components")
    axes[1].set_title("Residual exploitation")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_calibration(candidates: pd.DataFrame, output: Path) -> None:
    data = candidates.copy()
    if len(data) > 8000:
        data = data.sample(8000, random_state=1729)
    colors = {"faithful": "#2f7f6f", "view_impostor": "#b23a48", "drift": "#667085"}
    fig, ax = plt.subplots(figsize=(5.3, 4.2), constrained_layout=True)
    for mode, group in data.groupby("candidate_mode"):
        ax.scatter(group.proxy_score, group.true_iou, s=7, alpha=0.20, color=colors.get(mode, "#333333"), label=mode)
    ax.set_xlabel("candidate sparse-view proxy score")
    ax.set_ylabel("candidate true 3D IoU")
    ax.set_title("Proxy is weakly calibrated")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_failure_slices(failures: pd.DataFrame, output: Path) -> None:
    data = failures.head(12).copy()
    labels = [str(i + 1) for i in range(len(data))]
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8.5, 3.4), constrained_layout=True)
    ax.bar(x - 0.2, data.hidden_error, width=0.4, color="#b23a48", label="hidden error")
    ax.bar(x + 0.2, data.surface_fraction, width=0.4, color="#667085", label="surface fraction")
    ax.set_xticks(x, labels)
    ax.set_xlabel("worst repaired selections at N=256")
    ax.set_ylabel("diagnostic value")
    ax.set_title("Where the repair still fails")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _candidate_correlation(candidates: pd.DataFrame) -> pd.DataFrame:
    def corr_or_nan(group: pd.DataFrame, method: str) -> float:
        if group.proxy_score.nunique() <= 1 or group.true_iou.nunique() <= 1:
            return float("nan")
        return float(group.proxy_score.corr(group.true_iou, method=method))

    rows = []
    for mode, group in candidates.groupby("candidate_mode"):
        if len(group) < 3:
            continue
        rows.append(
            {
                "candidate_mode": mode,
                "count": int(len(group)),
                "pearson_proxy_true_iou": corr_or_nan(group, "pearson"),
                "spearman_proxy_true_iou": corr_or_nan(group, "spearman"),
                "proxy_score_mean": float(group.proxy_score.mean()),
                "true_iou_mean": float(group.true_iou.mean()),
            }
        )
    rows.append(
        {
            "candidate_mode": "all",
            "count": int(len(candidates)),
            "pearson_proxy_true_iou": corr_or_nan(candidates, "pearson"),
            "spearman_proxy_true_iou": corr_or_nan(candidates, "spearman"),
            "proxy_score_mean": float(candidates.proxy_score.mean()),
            "true_iou_mean": float(candidates.true_iou.mean()),
        }
    )
    return pd.DataFrame(rows)


def _claims(summary: pd.DataFrame, ablation: pd.DataFrame, corr: pd.DataFrame, mode: str) -> dict[str, object]:
    main = summary[(summary.regime == "candidate_count") & (summary.axis_label == "xy")]

    def get(method: str, n: int, column: str) -> float:
        row = main[(main.method == method) & (main.n == n)]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][column])

    naive_1_proxy = get("naive", 1, "proxy_score_mean")
    naive_256_proxy = get("naive", 256 if mode == "full-v3" else 8, "proxy_score_mean")
    naive_1_iou = get("naive", 1, "true_iou_mean")
    naive_256_iou = get("naive", 256 if mode == "full-v3" else 8, "true_iou_mean")
    naive_256_gap = get("naive", 256 if mode == "full-v3" else 8, "exploitation_gap_mean")
    repair_256_iou = get("repaired", 256 if mode == "full-v3" else 8, "true_iou_mean")
    repair_256_gap = get("repaired", 256 if mode == "full-v3" else 8, "exploitation_gap_mean")
    random_256_iou = get("random", 256 if mode == "full-v3" else 8, "true_iou_mean")

    baseline_path = ROOT / "results" / "full" / "summary.csv"
    baseline_numbers: dict[str, float] = {}
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)

        def bget(method: str, n: int, column: str) -> float:
            row = baseline[(baseline.method == method) & (baseline.n == n)]
            if row.empty:
                return float("nan")
            return float(row.iloc[0][column])

        baseline_numbers = {
            "baseline_naive_n1_proxy": bget("naive", 1, "proxy_score_mean"),
            "baseline_naive_n64_proxy": bget("naive", 64, "proxy_score_mean"),
            "baseline_naive_n1_true_iou": bget("naive", 1, "true_iou_mean"),
            "baseline_naive_n64_true_iou": bget("naive", 64, "true_iou_mean"),
            "baseline_naive_n64_gap": bget("naive", 64, "exploitation_gap_mean"),
            "baseline_repaired_n64_true_iou": bget("repaired", 64, "true_iou_mean"),
            "baseline_repaired_n64_gap": bget("repaired", 64, "exploitation_gap_mean"),
        }
    else:
        baseline_numbers = {
            "baseline_naive_n1_proxy": naive_1_proxy,
            "baseline_naive_n64_proxy": naive_256_proxy,
            "baseline_naive_n1_true_iou": naive_1_iou,
            "baseline_naive_n64_true_iou": naive_256_iou,
            "baseline_naive_n64_gap": naive_256_gap,
            "baseline_repaired_n64_true_iou": repair_256_iou,
            "baseline_repaired_n64_gap": repair_256_gap,
        }

    checks = {
        "replicated_baseline_proxy_increases": baseline_numbers["baseline_naive_n64_proxy"]
        >= baseline_numbers["baseline_naive_n1_proxy"] + (0.12 if mode == "full-v3" else 0.02),
        "replicated_baseline_true_iou_drops": baseline_numbers["baseline_naive_n64_true_iou"]
        <= baseline_numbers["baseline_naive_n1_true_iou"] - (0.08 if mode == "full-v3" else -0.05),
        "n256_proxy_saturates": naive_256_proxy >= 0.98,
        "n256_repair_beats_naive_true_iou": repair_256_iou >= naive_256_iou + (0.20 if mode == "full-v3" else 0.02),
        "n256_repair_reduces_gap": repair_256_gap <= naive_256_gap - (0.15 if mode == "full-v3" else 0.01),
        "n256_repair_beats_random": repair_256_iou >= random_256_iou,
    }
    corr_all = corr[corr.candidate_mode == "all"].iloc[0]
    checks["candidate_proxy_not_perfectly_calibrated"] = abs(float(corr_all.pearson_proxy_true_iou)) < 0.80

    return {
        "mode": mode,
        "claim_pass": bool(all(checks.values())),
        "checks": checks,
        "key_numbers": {
            "naive_n1_proxy": naive_1_proxy,
            "naive_nmax_proxy": naive_256_proxy,
            "naive_n1_true_iou": naive_1_iou,
            "naive_nmax_true_iou": naive_256_iou,
            "naive_nmax_gap": naive_256_gap,
            "repaired_nmax_true_iou": repair_256_iou,
            "repaired_nmax_gap": repair_256_gap,
            "random_nmax_true_iou": random_256_iou,
            **baseline_numbers,
        },
    }


def run_suite(config: SuiteConfig, output: Path) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, float | int | str]] = []
    all_candidate_rows: list[dict[str, float | int | str]] = []

    rows, candidate_rows = _run_regime(
        regime="candidate_count",
        grid=config.main_grid,
        scenes=config.main_scenes,
        seeds=config.main_seeds,
        n_values=config.main_n_values,
        axes=BASE_AXES,
        sampler=SamplerConfig(),
        sampler_label="base",
        methods=("naive", "repaired", "random"),
        collect_candidates=True,
    )
    all_rows.extend(rows)
    all_candidate_rows.extend(candidate_rows)

    for axes in [("x",), ("x", "y"), ("x", "y", "z")]:
        rows, _ = _run_regime(
            regime="axis_sweep",
            grid=config.stress_grid,
            scenes=config.stress_scenes,
            seeds=config.stress_seeds,
            n_values=config.stress_n_values,
            axes=axes,
            sampler=SamplerConfig(),
            sampler_label="base",
            methods=("naive", "repaired"),
        )
        all_rows.extend(rows)

    for impostor_prior in [0.10, 0.29, 0.45, 0.60]:
        sampler = SamplerConfig(
            faithful_prob=max(0.05, 0.85 - impostor_prior),
            view_impostor_prob=impostor_prior,
            drift_prob=0.15,
        )
        rows, _ = _run_regime(
            regime="mixture_stress",
            grid=config.stress_grid,
            scenes=config.stress_scenes,
            seeds=config.stress_seeds,
            n_values=config.stress_n_values,
            axes=BASE_AXES,
            sampler=sampler,
            sampler_label=f"impostor_{impostor_prior:.2f}",
            methods=("naive", "repaired"),
            extra={"impostor_prior": impostor_prior},
        )
        all_rows.extend(rows)

    for fill_scale in [0.40, 1.00, 2.00]:
        sampler = SamplerConfig(hidden_fill_scale=fill_scale)
        rows, _ = _run_regime(
            regime="hidden_mass_stress",
            grid=config.stress_grid,
            scenes=max(1, config.stress_scenes // 2),
            seeds=config.stress_seeds[: max(1, min(2, len(config.stress_seeds)))],
            n_values=config.stress_n_values[-2:],
            axes=BASE_AXES,
            sampler=sampler,
            sampler_label=f"fill_{fill_scale:.2f}",
            methods=("naive", "repaired"),
            extra={"hidden_fill_scale": fill_scale},
        )
        all_rows.extend(rows)

    rows, _ = _run_regime(
        regime="repair_ablation",
        grid=config.stress_grid,
        scenes=config.stress_scenes,
        seeds=config.stress_seeds,
        n_values=config.stress_n_values[-2:],
        axes=BASE_AXES,
        sampler=SamplerConfig(),
        sampler_label="base",
        methods=(
            "naive",
            "repaired",
            "no_consensus",
            "no_volume",
            "no_surface_penalty",
            "consensus_only",
            "volume_surface_only",
            "random",
        ),
    )
    all_rows.extend(rows)

    trials = pd.DataFrame(all_rows)
    candidates = pd.DataFrame(all_candidate_rows)
    group_cols = ["regime", "axis_label", "sampler_label", "method", "n"]
    for optional in ["impostor_prior", "hidden_fill_scale"]:
        if optional in trials.columns:
            group_cols.append(optional)
    summary = _summarize(trials, group_cols)
    ablation = _summarize(
        trials[trials.regime == "repair_ablation"],
        ["method", "n"],
    )
    corr = _candidate_correlation(candidates)
    failures = (
        trials[(trials.regime == "candidate_count") & (trials.method == "repaired") & (trials.n == max(config.main_n_values))]
        .sort_values(["true_iou", "hidden_error"], ascending=[True, False])
        .head(30)
        .reset_index(drop=True)
    )

    trials_path = output / "expanded_trials.csv"
    summary_path = output / "expanded_summary.csv"
    candidates_path = output / "candidate_diagnostics.csv"
    ablation_path = output / "ablation_summary.csv"
    corr_path = output / "correlation_summary.csv"
    failures_path = output / "failure_cases.csv"
    manifest_path = output / "manifest.json"
    claims_path = output / "claims.json"

    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    ablation.to_csv(ablation_path, index=False)
    corr.to_csv(corr_path, index=False)
    failures.to_csv(failures_path, index=False)

    _plot_candidate_count(summary, figures / "figure4_candidate_count_256.png")
    _plot_axis_sweep(summary, figures / "figure5_axis_sweep.png")
    _plot_mixture_stress(summary, figures / "figure6_mixture_stress.png")
    _plot_repair_ablation(ablation, figures / "figure7_repair_ablation.png")
    _plot_calibration(candidates, figures / "figure8_proxy_calibration.png")
    _plot_failure_slices(failures, figures / "figure9_failure_slices.png")

    for figure in [
        "figure4_candidate_count_256.png",
        "figure5_axis_sweep.png",
        "figure6_mixture_stress.png",
        "figure7_repair_ablation.png",
        "figure8_proxy_calibration.png",
        "figure9_failure_slices.png",
    ]:
        target = output / figure
        source = figures / figure
        if source.exists():
            target.write_bytes(source.read_bytes())

    claims = _claims(summary, ablation, corr, config.mode)
    claims_path.write_text(json.dumps(claims, indent=2), encoding="utf-8")
    manifest = {
        "mode": config.mode,
        "main_grid": config.main_grid,
        "main_scenes": config.main_scenes,
        "main_seeds": list(config.main_seeds),
        "main_n_values": list(config.main_n_values),
        "stress_grid": config.stress_grid,
        "stress_scenes": config.stress_scenes,
        "stress_seeds": list(config.stress_seeds),
        "stress_n_values": list(config.stress_n_values),
        "outputs": {
            "trials": str(trials_path),
            "summary": str(summary_path),
            "candidates": str(candidates_path),
            "ablation": str(ablation_path),
            "correlation": str(corr_path),
            "failures": str(failures_path),
            "claims": str(claims_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "trials": trials_path,
        "summary": summary_path,
        "candidates": candidates_path,
        "ablation": ablation_path,
        "correlation": corr_path,
        "failures": failures_path,
        "claims": claims_path,
        "manifest": manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v3 view-impostor expansion suite.")
    parser.add_argument("--mode", choices=sorted(PRESETS), default="full")
    parser.add_argument("--output", type=Path, default=Path("results/expansion"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_suite(PRESETS[args.mode], args.output)
    claims = json.loads(paths["claims"].read_text(encoding="utf-8"))
    print(f"expansion suite complete: {claims['mode']}; claim_pass={claims['claim_pass']}")
    for name, path in paths.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
