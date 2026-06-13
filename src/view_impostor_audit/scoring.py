"""Proxy and uncertainty-aware scoring rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from view_impostor_audit.geometry import (
    front_depth,
    iou,
    normalize_axes,
    project_silhouette,
    surface_fraction,
    visual_hull_from_scene,
)
from view_impostor_audit.scene import Candidate


@dataclass(frozen=True)
class CandidateScore:
    proxy: float
    consensus: float
    volume_plausibility: float
    surface_penalty: float
    repair: float


def proxy_score(candidate: np.ndarray, target: np.ndarray, axes: Iterable[str] = ("x", "y")) -> float:
    """Sparse-view score based on silhouettes and first-hit depth."""

    axes = normalize_axes(axes)
    values: list[float] = []
    for axis in axes:
        cand_sil = project_silhouette(candidate, axis)
        targ_sil = project_silhouette(target, axis)
        sil_score = iou(cand_sil, targ_sil)

        cand_depth, cand_hit = front_depth(candidate, axis)
        targ_depth, targ_hit = front_depth(target, axis)
        union = cand_hit | targ_hit
        if union.any():
            depth_score = 1.0 - float(np.abs(cand_depth[union] - targ_depth[union]).mean())
        else:
            depth_score = 1.0
        values.append(0.70 * sil_score + 0.30 * max(0.0, min(1.0, depth_score)))
    return float(np.mean(values))


def _ensemble_consensus(candidates: Sequence[Candidate]) -> np.ndarray:
    stack = np.stack([candidate.occupancy for candidate in candidates], axis=0)
    # A low threshold keeps consensus meaningful for small candidate sets while
    # still downweighting idiosyncratic hidden hallucinations.
    threshold = min(0.50, max(1.0 / max(len(candidates), 1), 0.32))
    return stack.mean(axis=0) >= threshold


def _volume_plausibility(candidate: np.ndarray, target: np.ndarray, axes: Iterable[str]) -> float:
    hull = visual_hull_from_scene(target, axes)
    visible_upper = max(float(hull.sum()), 1.0)
    visible_lower = max(float(sum(project_silhouette(target, axis).sum() for axis in axes)), 1.0)
    expected = max(0.44 * visible_upper, 1.15 * visible_lower)
    actual = float(candidate.sum())
    rel = abs(actual - expected) / max(expected, 1.0)
    return float(np.exp(-2.0 * rel))


def score_candidates(
    candidates: Sequence[Candidate],
    target: np.ndarray,
    axes: Iterable[str] = ("x", "y"),
) -> list[CandidateScore]:
    """Score candidates using both sparse proxy and repair features."""

    axes = normalize_axes(axes)
    consensus = _ensemble_consensus(candidates)
    scores: list[CandidateScore] = []
    for candidate in candidates:
        proxy = proxy_score(candidate.occupancy, target, axes)
        consensus_score = iou(candidate.occupancy, consensus)
        plausibility = _volume_plausibility(candidate.occupancy, target, axes)
        surface = surface_fraction(candidate.occupancy)
        repair = 0.58 * proxy + 0.27 * consensus_score + 0.22 * plausibility - 0.12 * surface
        scores.append(
            CandidateScore(
                proxy=proxy,
                consensus=consensus_score,
                volume_plausibility=plausibility,
                surface_penalty=surface,
                repair=float(repair),
            )
        )
    return scores


def select_index(scores: Sequence[CandidateScore], method: str = "naive") -> int:
    if method == "naive":
        values = [score.proxy for score in scores]
    elif method == "repaired":
        values = [score.repair for score in scores]
    else:
        raise ValueError(f"unknown selection method: {method}")
    return int(np.argmax(values))
