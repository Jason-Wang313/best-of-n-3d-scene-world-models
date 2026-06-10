"""Synthetic scene generation and simulated 3D world-model candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from bon3d.geometry import (
    behind_front_mask,
    dilate_once,
    normalize_axes,
    shift_occupancy,
    visible_mask,
    visual_hull_from_scene,
)


@dataclass(frozen=True)
class Scene:
    occupancy: np.ndarray
    seed: int
    primitive_count: int


@dataclass(frozen=True)
class Candidate:
    occupancy: np.ndarray
    mode: str
    index: int


def _add_box(occupancy: np.ndarray, rng: np.random.Generator) -> None:
    n = occupancy.shape[0]
    lo = rng.integers(2, max(3, n // 2), size=3)
    size = rng.integers(max(3, n // 8), max(4, n // 3), size=3)
    hi = np.minimum(lo + size, n - 2)
    occupancy[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True


def _add_ellipsoid(occupancy: np.ndarray, rng: np.random.Generator) -> None:
    n = occupancy.shape[0]
    center = rng.uniform(n * 0.25, n * 0.75, size=3)
    radius = rng.uniform(n * 0.10, n * 0.22, size=3)
    x, y, z = np.indices(occupancy.shape)
    value = ((x - center[0]) / radius[0]) ** 2
    value += ((y - center[1]) / radius[1]) ** 2
    value += ((z - center[2]) / radius[2]) ** 2
    occupancy[value <= 1.0] = True


def generate_scene(rng: np.random.Generator, grid: int = 24, seed: int | None = None) -> Scene:
    """Create a compact solid scene made from boxes and ellipsoids."""

    for attempt in range(50):
        occ = np.zeros((grid, grid, grid), dtype=bool)
        primitive_count = int(rng.integers(3, 6))
        for _ in range(primitive_count):
            if rng.random() < 0.58:
                _add_box(occ, rng)
            else:
                _add_ellipsoid(occ, rng)
        frac = occ.mean()
        if 0.035 <= frac <= 0.32:
            return Scene(occ, seed=-1 if seed is None else seed, primitive_count=primitive_count)
    return Scene(occ, seed=-1 if seed is None else seed, primitive_count=primitive_count)


def _apply_voxel_noise(
    occupancy: np.ndarray,
    rng: np.random.Generator,
    delete_prob: float,
    add_prob: float,
    add_pool: np.ndarray | None = None,
) -> np.ndarray:
    occ = np.asarray(occupancy, dtype=bool).copy()
    if occ.any() and delete_prob > 0:
        delete = rng.random(occ.shape) < delete_prob
        occ &= ~delete
    if add_prob > 0:
        if add_pool is None:
            add_pool = dilate_once(occ) & ~occ
        add = (rng.random(occ.shape) < add_prob) & add_pool
        occ |= add
    return occ


def _faithful_candidate(target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shift = rng.choice([-1, 0, 1], size=3, p=[0.18, 0.64, 0.18])
    occ = shift_occupancy(target, shift)
    nearby = dilate_once(target) & ~target
    occ = _apply_voxel_noise(occ, rng, delete_prob=0.030, add_prob=0.028, add_pool=nearby)
    if rng.random() < 0.25:
        occ = dilate_once(occ) if rng.random() < 0.5 else (occ & ~((dilate_once(~occ)) & occ & (rng.random(occ.shape) < 0.10)))
    return occ


def _drift_candidate(target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shift = rng.integers(-3, 4, size=3)
    occ = shift_occupancy(target, shift)
    hull = visual_hull_from_scene(target, ("x", "y", "z"))
    occ = _apply_voxel_noise(occ, rng, delete_prob=0.16, add_prob=0.018, add_pool=hull & ~occ)
    return occ


def _view_impostor_candidate(target: np.ndarray, rng: np.random.Generator, axes: Iterable[str]) -> np.ndarray:
    axes = normalize_axes(axes)
    observed = visible_mask(target, axes)
    hull = visual_hull_from_scene(target, axes)
    hidden_pool = behind_front_mask(target, axes) & hull & ~observed
    occ = observed.copy()

    # Most impostors are thin shells; some carry random hidden mass while still
    # preserving every sparse-view first hit and silhouette.
    if rng.random() < 0.72:
        fill_prob = rng.uniform(0.000, 0.018)
    else:
        fill_prob = rng.uniform(0.025, 0.070)
    occ |= (rng.random(target.shape) < fill_prob) & hidden_pool

    # Rarely add a small coherent false block inside the visual hull.
    if rng.random() < 0.30 and hidden_pool.any():
        coords = np.argwhere(hidden_pool)
        center = coords[int(rng.integers(0, len(coords)))]
        radius = int(rng.integers(2, 4))
        x, y, z = np.indices(target.shape)
        block = (
            (np.abs(x - center[0]) <= radius)
            & (np.abs(y - center[1]) <= radius)
            & (np.abs(z - center[2]) <= radius)
            & hidden_pool
        )
        occ |= block
    return occ


def sample_candidates(
    target: np.ndarray,
    n: int,
    rng: np.random.Generator,
    proxy_axes: Iterable[str] = ("x", "y"),
) -> list[Candidate]:
    """Sample candidate next scenes from a biased synthetic world model."""

    candidates: list[Candidate] = []
    for index in range(n):
        mode = str(rng.choice(["faithful", "view_impostor", "drift"], p=[0.56, 0.29, 0.15]))
        if mode == "faithful":
            occ = _faithful_candidate(target, rng)
        elif mode == "view_impostor":
            occ = _view_impostor_candidate(target, rng, proxy_axes)
        else:
            occ = _drift_candidate(target, rng)
        candidates.append(Candidate(occ.astype(bool), mode=mode, index=index))
    return candidates
