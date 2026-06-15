"""Run the ModelNet10 tiny real-shape view-impostor benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from view_impostor_audit.modelnet_benchmark import (  # noqa: E402
    make_figure,
    run_modelnet_benchmark,
    summarize,
    write_outputs,
)


def run(quick: bool = False, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path.cwd()
    artifact_dir = Path(output_dir) if output_dir is not None else root / "results" / "modelnet10_benchmark"
    data_dir = root / "data" / "modelnet10_tiny"
    figure_path = (
        artifact_dir / "figures" / "figure10_modelnet10_benchmark.png"
        if output_dir is not None
        else root / "figures" / "figure10_modelnet10_benchmark.png"
    )
    per_category = 2 if quick else 8
    rows, meta = run_modelnet_benchmark(data_dir=data_dir, per_category=per_category, seed=2026)
    summary = summarize(rows)
    make_figure(summary, figure_path)
    return write_outputs(rows, meta, artifact_dir, figure_path, quick=quick)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ModelNet10 tiny benchmark.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = run(quick=args.quick, output_dir=args.output)
    claims = json.loads(Path(payload["claims"]).read_text(encoding="utf-8"))
    print(f"ModelNet10 benchmark: {claims['summary']}")
    print(f"all_passed={claims['all_passed']}")
    print(f"Manifest: {payload['manifest']}")
    return 0 if claims["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
