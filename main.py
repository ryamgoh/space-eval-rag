from __future__ import annotations

import argparse

from inference.evaluator import LLMEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiple LLMs on multiple tasks.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    evaluator = LLMEvaluator(args.config)
    evaluator.run()


if __name__ == "__main__":
    main()
