from __future__ import annotations

import json
from pathlib import Path

from experiments.run_modelnet10_benchmark import run


def test_modelnet10_benchmark_quick_writes_passing_claims(tmp_path: Path) -> None:
    manifest = run(quick=True, output_dir=tmp_path / "modelnet10")

    for key in ("trials", "summary", "claims", "figure", "manifest"):
        assert Path(manifest[key]).exists()

    claims = json.loads(Path(manifest["claims"]).read_text(encoding="utf-8"))
    assert claims["all_passed"]
    assert "modelnet10_hidden_iou_drops" in claims["checks"]
