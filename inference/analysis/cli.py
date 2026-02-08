from __future__ import annotations

import argparse
from pathlib import Path

from inference.analysis.runner import AnalysisManager
from inference.config.manager import ConfigManager


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate analysis plots for runs.")
    parser.add_argument("--runs", nargs="+", help="Run directories or run IDs.")
    parser.add_argument("--out", help="Output directory for comparison plots.")
    parser.add_argument("--config", help="Optional config file with analysis settings.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = None
    if args.config:
        config = ConfigManager.load_yaml(args.config)
    manager = AnalysisManager((config or {}).get("analysis"))

    if args.runs:
        manager.compare_runs(args.runs, out_dir=args.out)
        return

    if config and config.get("analysis", {}).get("compare_runs"):
        compare_runs = config["analysis"]["compare_runs"]
        manager.compare_runs(compare_runs, out_dir=args.out)
        return

    raise SystemExit("No runs specified. Use --runs or provide analysis.compare_runs in config.")


if __name__ == "__main__":
    main()
