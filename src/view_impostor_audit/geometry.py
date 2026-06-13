"""Voxel geometry utilities used by the synthetic scene benchmark."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

AXES: tuple[str, str, str] = ("x", "y", "z")
AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def normalize_axes(axes: Iterable[str]) -> tuple[str, ...]:
    result = tuple(axes)
    unknown = [axis for axis in result if axis not in AXIS_TO_INDEX]
    if unknown:
        raise ValueError(f"unknown view axes: {unknown}")
    return result


def project_silhouette(occupancy: np.ndarray, axis: str) -> np.ndarray:
    """Return the binary silhouette seen from the negative side of an axis."""

    return np.asarray(occupancy, dtype=bool).any(axis=AXIS_TO_INDEX[axis])


def front_index(occupancy: np.ndarray, axis: str) -> tuple[np.ndarray, np.ndarray]:
    """Return first occupied voxel index and a per-ray hit mask."""

    occ = np.asarray(occupancy, dtype=bool)
    idx = AXIS_TO_INDEX[axis]
    moved = np.moveaxis(occ, idx, 0)
    hit = moved.any(axis=0)
    first = np.argmax(moved, axis=0).astype(np.int16)
    first = np.where(hit, first, occ.shape[idx]).astype(np.int16)
    return first, hit


def front_depth(occupancy: np.ndarray, axis: str) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized first-hit depth and a per-ray hit mask."""

    first, hit = front_index(occupancy, axis)
    denom = max(np.asarray(occupancy).shape[AXIS_TO_INDEX[axis]] - 1, 1)
    depth = np.where(hit, first / denom, 1.0).astype(float)
    return depth, hit


def visible_mask(occupancy: np.ndarray, axes: Iterable[str]) -> np.ndarray:
    """Voxels that are first hits from at least one supplied view axis."""

    axes = normalize_axes(axes)
    occ = np.asarray(occupancy, dtype=bool)
    mask = np.zeros_like(occ, dtype=bool)
    for axis in axes:
        idx = AXIS_TO_INDEX[axis]
        moved = np.moveaxis(occ, idx, 0)
        hit = moved.any(axis=0)
        first = np.argmax(moved, axis=0)
        local = np.zeros_like(moved, dtype=bool)
        coords = np.nonzero(hit)
        if coords[0].size:
            local[(first[coords],) + coords] = True
        mask |= np.moveaxis(local, 0, idx)
    return mask


def visual_hull_from_scene(occupancy: np.ndarray, axes: Iterable[str]) -> np.ndarray:
    """Visual hull induced by silhouettes from the given axes."""

    axes = normalize_axes(axes)
    occ = np.asarray(occupancy, dtype=bool)
    hull = np.ones_like(occ, dtype=bool)
    for axis in axes:
        sil = project_silhouette(occ, axis)
        if axis == "x":
            hull &= sil[None, :, :]
        elif axis == "y":
            hull &= sil[:, None, :]
        elif axis == "z":
            hull &= sil[:, :, None]
    return hull


def behind_front_mask(occupancy: np.ndarray, axes: Iterable[str]) -> np.ndarray:
    """Voxels that lie behind the first observed surface for all view axes."""

    axes = normalize_axes(axes)
    occ = np.asarray(occupancy, dtype=bool)
    n = occ.shape[0]
    mask = visual_hull_from_scene(occ, axes)
    coords = np.arange(n)
    for axis in axes:
        first, hit = front_index(occ, axis)
        if axis == "x":
            mask &= hit[None, :, :]
            mask &= coords[:, None, None] >= first[None, :, :]
        elif axis == "y":
            mask &= hit[:, None, :]
            mask &= coords[None, :, None] >= first[:, None, :]
        elif axis == "z":
            mask &= hit[:, :, None]
            mask &= coords[None, None, :] >= first[:, :, None]
    return mask


def iou(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Intersection over union for binary voxel arrays."""

    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        aa = aa & m
        bb = bb & m
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(aa, bb).sum() / union)


def shift_occupancy(occupancy: np.ndarray, shift: Sequence[int]) -> np.ndarray:
    """Translate an occupancy grid with zero padding instead of wraparound."""

    occ = np.asarray(occupancy, dtype=bool)
    out = np.zeros_like(occ, dtype=bool)
    src_slices = []
    dst_slices = []
    for size, delta in zip(occ.shape, shift):
        if delta >= 0:
            src_slices.append(slice(0, size - delta))
            dst_slices.append(slice(delta, size))
        else:
            src_slices.append(slice(-delta, size))
            dst_slices.append(slice(0, size + delta))
    out[tuple(dst_slices)] = occ[tuple(src_slices)]
    return out


def six_neighbor_count(occupancy: np.ndarray) -> np.ndarray:
    occ = np.asarray(occupancy, dtype=bool)
    padded = np.pad(occ.astype(np.int8), 1)
    core = padded[1:-1, 1:-1, 1:-1]
    return (
        core
        + padded[:-2, 1:-1, 1:-1]
        + padded[2:, 1:-1, 1:-1]
        + padded[1:-1, :-2, 1:-1]
        + padded[1:-1, 2:, 1:-1]
        + padded[1:-1, 1:-1, :-2]
        + padded[1:-1, 1:-1, 2:]
    )


def dilate_once(occupancy: np.ndarray) -> np.ndarray:
    return six_neighbor_count(occupancy) > 0


def erode_once(occupancy: np.ndarray) -> np.ndarray:
    return six_neighbor_count(occupancy) >= 7


def surface_fraction(occupancy: np.ndarray) -> float:
    """Fraction of occupied voxels exposed to empty space."""

    occ = np.asarray(occupancy, dtype=bool)
    count = int(occ.sum())
    if count == 0:
        return 1.0
    interior = erode_once(occ) & occ
    surface = occ & ~interior
    return float(surface.sum() / count)


def pairwise_diversity(occupancies: Sequence[np.ndarray]) -> float:
    """Mean one-minus-IoU across candidate pairs."""

    if len(occupancies) < 2:
        return 0.0
    values = [1.0 - iou(occupancies[i], occupancies[j]) for i, j in combinations(range(len(occupancies)), 2)]
    return float(np.mean(values))
