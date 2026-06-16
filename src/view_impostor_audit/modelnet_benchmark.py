"""CPU-light ModelNet10 real-shape benchmark for view-impostor audits."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from view_impostor_audit.metrics import evaluate_selection
from view_impostor_audit.scene import SamplerConfig, sample_candidates
from view_impostor_audit.scoring import score_candidates, select_index


CATEGORIES = ("chair", "monitor")
N_VALUES = (1, 4, 16, 64)
MAX_N = 64
PROXY_AXES = ("x", "y")
SAMPLER = SamplerConfig(
    faithful_prob=0.85,
    view_impostor_prob=0.12,
    drift_prob=0.03,
    hidden_fill_scale=0.50,
    coherent_block_prob=0.20,
    thin_shell_prob=0.90,
)


def load_modelnet_slice(data_dir: Path, *, per_category: int = 8) -> list[dict[str, Any]]:
    """Load deterministic real-shape targets from the committed ModelNet10 slice."""

    records: list[dict[str, Any]] = []
    for category in CATEGORIES:
        path = data_dir / f"{category}.npy.gz"
        with gzip.open(path, "rb") as handle:
            array = np.load(handle, allow_pickle=False)
        indices = np.linspace(0, array.shape[0] - 1, int(per_category), dtype=int)
        for local_idx, source_idx in enumerate(indices):
            occupancy = np.asarray(array[int(source_idx), 0] > 0.5, dtype=bool)
            records.append(
                {
                    "category": category,
                    "local_idx": int(local_idx),
                    "source_idx": int(source_idx),
                    "occupancy": occupancy,
                }
            )
    return records


def run_modelnet_benchmark(
    *,
    data_dir: Path,
    per_category: int = 8,
    seed: int = 2026,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shapes = load_modelnet_slice(data_dir, per_category=per_category)
    for shape_id, record in enumerate(shapes):
        target = record["occupancy"]
        source_idx = int(record["source_idx"])
        rng = np.random.default_rng(seed + 1000 * shape_id + source_idx)
        pool = sample_candidates(target, MAX_N, rng, proxy_axes=PROXY_AXES, sampler=SAMPLER)
        for n_value in N_VALUES:
            candidates = pool[:n_value]
            scores = score_candidates(candidates, target, axes=PROXY_AXES)
            for method in ("naive", "repaired"):
                selected = select_index(scores, method)
                row = evaluate_selection(
                    target,
                    candidates,
                    scores,
                    selected,
                    PROXY_AXES,
                    method,
                    n_value,
                    scene_id=shape_id,
                )
                row.update(
                    {
                        "benchmark": "ModelNet10 tiny slice",
                        "category": record["category"],
                        "source_idx": source_idx,
                        "local_idx": int(record["local_idx"]),
                    }
                )
                rows.append(row)
    meta = {
        "benchmark": "ModelNet10 tiny slice",
        "categories": list(CATEGORIES),
        "per_category": int(per_category),
        "shape_count": len(shapes),
        "candidate_counts": list(N_VALUES),
        "seed": int(seed),
        "proxy_axes": list(PROXY_AXES),
        "sampler": SAMPLER.__dict__,
    }
    return rows, meta


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in rows})
    n_values = sorted({int(row["n"]) for row in rows})
    for method in methods:
        for n_value in n_values:
            group = [row for row in rows if row["method"] == method and int(row["n"]) == n_value]
            if not group:
                continue
            item: dict[str, Any] = {"method": method, "n": n_value, "count": len(group)}
            for key in (
                "proxy_score",
                "true_iou",
                "hidden_iou",
                "hidden_error",
                "exploitation_gap",
                "candidate_diversity",
                "top_proxy_diversity",
                "surface_fraction",
                "visible_occupancy_fraction",
            ):
                item[f"{key}_mean"] = float(np.mean([float(row[key]) for row in group]))
            item["view_impostor_rate"] = float(
                np.mean([str(row["selected_mode"]) == "view_impostor" for row in group])
            )
            summary.append(item)
    return summary


def claim_gates(summary: list[dict[str, Any]], *, quick: bool = False) -> dict[str, Any]:
    def row(method: str, n_value: int) -> dict[str, Any]:
        matches = [item for item in summary if item["method"] == method and int(item["n"]) == n_value]
        if not matches:
            raise KeyError((method, n_value))
        return matches[0]

    naive_1 = row("naive", 1)
    naive_64 = row("naive", 64)
    repaired_64 = row("repaired", 64)
    thresholds = {
        "proxy": 0.04 if quick else 0.10,
        "true_iou_drop": -0.02 if quick else -0.08,
        "hidden_iou_drop": -0.15 if quick else -0.30,
        "repair_iou": 0.12 if quick else 0.25,
        "gap_reduction": 0.12 if quick else 0.25,
        "impostor_rate": 0.49 if quick else 0.65,
    }
    checks = {
        "modelnet10_proxy_rises": _claim(
            naive_64["proxy_score_mean"] - naive_1["proxy_score_mean"],
            thresholds["proxy"],
            ">",
            "Increasing candidate count raises sparse-view proxy fit on ModelNet10 shapes.",
        ),
        "modelnet10_true_iou_drops": _claim(
            naive_64["true_iou_mean"] - naive_1["true_iou_mean"],
            thresholds["true_iou_drop"],
            "<",
            "The same raw selector reduces true 3D IoU.",
        ),
        "modelnet10_hidden_iou_drops": _claim(
            naive_64["hidden_iou_mean"] - naive_1["hidden_iou_mean"],
            thresholds["hidden_iou_drop"],
            "<",
            "The hidden-region IoU collapses under raw high-N sparse-view selection.",
        ),
        "modelnet10_repair_recovers_true_iou": _claim(
            repaired_64["true_iou_mean"] - naive_64["true_iou_mean"],
            thresholds["repair_iou"],
            ">",
            "Coverage-aware reranking recovers true 3D IoU over raw high-N selection.",
        ),
        "modelnet10_repair_reduces_gap": _claim(
            naive_64["exploitation_gap_mean"] - repaired_64["exploitation_gap_mean"],
            thresholds["gap_reduction"],
            ">",
            "Coverage-aware reranking reduces proxy-versus-true exploitation gap.",
        ),
        "modelnet10_high_n_selects_impostors": _claim(
            naive_64["view_impostor_rate"],
            thresholds["impostor_rate"],
            ">",
            "Raw high-N sparse-view selection is dominated by view-impostor candidates.",
        ),
    }
    return {
        "all_passed": all(payload["passed"] for payload in checks.values()),
        "checks": checks,
        "summary": (
            f"proxy change {naive_64['proxy_score_mean'] - naive_1['proxy_score_mean']:.3f}, "
            f"true-IoU change {naive_64['true_iou_mean'] - naive_1['true_iou_mean']:.3f}, "
            f"repair gain {repaired_64['true_iou_mean'] - naive_64['true_iou_mean']:.3f}."
        ),
    }


def make_figure(summary: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), sharex=True)
    colors = {"naive": "#d95f02", "repaired": "#1b9e77"}
    labels = {"naive": "sparse-view selector", "repaired": "coverage reranker"}
    for method in ("naive", "repaired"):
        rows = sorted([row for row in summary if row["method"] == method], key=lambda item: int(item["n"]))
        n_values = [int(row["n"]) for row in rows]
        axes[0].plot(n_values, [row["proxy_score_mean"] for row in rows], marker="o", color=colors[method], label=labels[method])
        axes[1].plot(n_values, [row["true_iou_mean"] for row in rows], marker="o", color=colors[method])
        axes[2].plot(n_values, [row["hidden_iou_mean"] for row in rows], marker="o", color=colors[method])
    axes[0].set_title("sparse-view proxy")
    axes[1].set_title("true 3D IoU")
    axes[2].set_title("hidden-region IoU")
    axes[0].set_ylabel("mean score")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_VALUES)
        ax.set_xticklabels([str(value) for value in N_VALUES])
        ax.set_xlabel("candidate count N")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("ModelNet10 real-shape view-impostor benchmark")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_outputs(rows: list[dict[str, Any]], meta: dict[str, Any], output_dir: Path, figure_path: Path, *, quick: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    claims = claim_gates(summary, quick=quick)
    trials_path = output_dir / "modelnet10_trials.csv"
    summary_path = output_dir / "modelnet10_summary.csv"
    claims_path = output_dir / "claims.json"
    manifest_path = output_dir / "manifest.json"
    _write_csv(trials_path, rows)
    _write_csv(summary_path, summary)
    _write_json(claims_path, claims)
    manifest = {
        **meta,
        "quick": bool(quick),
        "trials": str(trials_path),
        "summary": str(summary_path),
        "claims": str(claims_path),
        "figure": str(figure_path),
        "all_passed": claims["all_passed"],
    }
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}


def _claim(value: float, threshold: float, op: str, description: str) -> dict[str, Any]:
    passed = value > threshold if op == ">" else value < threshold
    return {
        "passed": bool(passed),
        "observed": float(value),
        "threshold": float(threshold),
        "op": op,
        "description": description,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key == "occupancy":
                continue
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
