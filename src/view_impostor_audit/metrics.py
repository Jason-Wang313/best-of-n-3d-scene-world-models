"""Evaluation diagnostics for selected 3D scene candidates."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from view_impostor_audit.geometry import iou, pairwise_diversity, surface_fraction, visible_mask, visual_hull_from_scene
from view_impostor_audit.scene import Candidate
from view_impostor_audit.scoring import CandidateScore


def hidden_region_mask(target: np.ndarray, axes: Iterable[str]) -> np.ndarray:
    return visual_hull_from_scene(target, axes) & ~visible_mask(target, axes)


def evaluate_selection(
    target: np.ndarray,
    candidates: Sequence[Candidate],
    scores: Sequence[CandidateScore],
    selected_index: int,
    axes: Iterable[str],
    method: str,
    n_value: int,
    scene_id: int,
) -> dict[str, float | int | str]:
    candidate = candidates[selected_index]
    score = scores[selected_index]
    hidden_mask = hidden_region_mask(target, axes)
    true_iou = iou(candidate.occupancy, target)
    hidden_iou = iou(candidate.occupancy, target, mask=hidden_mask)
    occ_count = max(float(candidate.occupancy.sum()), 1.0)
    visible_count = float((candidate.occupancy & visible_mask(candidate.occupancy, axes)).sum())
    top_k = min(5, len(candidates))
    order = np.argsort([-s.proxy for s in scores])[:top_k]
    top_diversity = pairwise_diversity([candidates[int(i)].occupancy for i in order])
    return {
        "scene_id": scene_id,
        "n": n_value,
        "method": method,
        "selected_index": selected_index,
        "selected_mode": candidate.mode,
        "proxy_score": score.proxy,
        "repair_score": score.repair,
        "true_iou": true_iou,
        "hidden_iou": hidden_iou,
        "hidden_error": 1.0 - hidden_iou,
        "exploitation_gap": score.proxy - true_iou,
        "candidate_diversity": pairwise_diversity([c.occupancy for c in candidates]),
        "top_proxy_diversity": top_diversity,
        "occupancy_fraction": float(candidate.occupancy.mean()),
        "surface_fraction": surface_fraction(candidate.occupancy),
        "visible_occupancy_fraction": visible_count / occ_count,
        "consensus_score": score.consensus,
        "volume_plausibility": score.volume_plausibility,
    }
