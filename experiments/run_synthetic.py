from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bon3d.experiment import PRESETS, ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic Best-of-N 3D scene benchmark.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--output", type=Path, default=Path("results/smoke"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenes", type=int, default=None)
    parser.add_argument("--grid", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = PRESETS[args.preset]
    config = ExperimentConfig(
        seed=base.seed if args.seed is None else args.seed,
        grid=base.grid if args.grid is None else args.grid,
        scenes=base.scenes if args.scenes is None else args.scenes,
        n_values=base.n_values,
        proxy_axes=base.proxy_axes,
    )
    paths = run_experiment(config, args.output)
    print(f"wrote summary: {paths['summary']}")
    print(f"wrote metrics: {paths['metrics']}")


if __name__ == "__main__":
    main()
