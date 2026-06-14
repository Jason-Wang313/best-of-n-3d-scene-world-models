from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.run_expansion_suite import PRESETS, run_suite
from view_impostor_audit.scene import SamplerConfig, generate_scene, sample_candidates


def test_sampler_config_changes_impostor_mix() -> None:
    scene = generate_scene(np.random.default_rng(11), grid=16)
    low = sample_candidates(
        scene.occupancy,
        120,
        np.random.default_rng(12),
        sampler=SamplerConfig(faithful_prob=0.85, view_impostor_prob=0.05, drift_prob=0.10),
    )
    high = sample_candidates(
        scene.occupancy,
        120,
        np.random.default_rng(12),
        sampler=SamplerConfig(faithful_prob=0.25, view_impostor_prob=0.65, drift_prob=0.10),
    )
    low_rate = sum(candidate.mode == "view_impostor" for candidate in low) / len(low)
    high_rate = sum(candidate.mode == "view_impostor" for candidate in high) / len(high)
    assert high_rate > low_rate + 0.30


def test_quick_expansion_suite_writes_claim_audit(tmp_path) -> None:
    paths = run_suite(PRESETS["quick"], tmp_path)
    for path in paths.values():
        assert path.exists()

    claims = json.loads(paths["claims"].read_text(encoding="utf-8"))
    assert claims["mode"] == "quick-smoke"
    assert "n256_repair_reduces_gap" in claims["checks"]

    summary = pd.read_csv(paths["summary"])
    assert {"candidate_count", "repair_ablation"} <= set(summary["regime"])
    assert {"naive", "repaired", "random"} <= set(summary["method"])
