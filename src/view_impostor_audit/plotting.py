"""Plot helpers for the synthetic experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from view_impostor_audit.geometry import project_silhouette


def plot_tradeoff(summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    colors = {"naive": "#b23a48", "repaired": "#2f7f6f"}
    labels = {"naive": "Sparse-view selector", "repaired": "Coverage-aware rerank"}
    for method, group in summary.groupby("method"):
        group = group.sort_values("n")
        x = group["n"].to_numpy()
        axes[0].plot(x, group["proxy_score_mean"], marker="o", color=colors[method], label=labels[method])
        axes[1].plot(x, group["true_iou_mean"], marker="o", color=colors[method])
        axes[2].plot(x, group["exploitation_gap_mean"], marker="o", color=colors[method])
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("candidate count N")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("proxy score")
    axes[1].set_ylabel("true 3D IoU")
    axes[2].set_ylabel("proxy - true IoU")
    axes[0].set_title("Sparse score rises")
    axes[1].set_title("3D consistency")
    axes[2].set_title("Exploitation gap")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_hidden_and_diversity(summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
    colors = {"naive": "#b23a48", "repaired": "#2f7f6f"}
    for method, group in summary.groupby("method"):
        group = group.sort_values("n")
        axes[0].plot(group["n"], group["hidden_error_mean"], marker="o", color=colors[method], label=method)
        axes[1].plot(group["n"], group["top_proxy_diversity_mean"], marker="o", color=colors[method], label=method)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("candidate count N")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("hidden-region error")
    axes[1].set_ylabel("top-proxy diversity")
    axes[0].set_title("Occluded geometry")
    axes[1].set_title("Diversity collapse")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_example(target: np.ndarray, naive: np.ndarray, repaired: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    scenes = [target, naive, repaired]
    titles = ["target", "naive selected", "repaired selected"]
    fig, axes = plt.subplots(3, 3, figsize=(6.9, 6.8), constrained_layout=True)
    for row, (scene, title) in enumerate(zip(scenes, titles)):
        for col, axis_name in enumerate(["x", "y", "z"]):
            ax = axes[row, col]
            ax.imshow(project_silhouette(scene, axis_name), cmap="magma", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{axis_name}-view")
            if col == 0:
                ax.set_ylabel(title)
    fig.savefig(output, dpi=220)
    plt.close(fig)
