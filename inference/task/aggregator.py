from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


class ResultAggregator:
    @staticmethod
    def aggregate_results(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        results_list = list(results)
        return {
            "runs": results_list,
            "summary": ResultAggregator.compute_statistics(results_list),
        }

    @staticmethod
    def compute_statistics(results: Iterable[Dict[str, Any]]) -> Dict[str, float]:
        metrics: Dict[str, List[float]] = {}
        for result in results:
            for name, value in result.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    metrics.setdefault(name, []).append(float(value))
        return {name: mean(values) for name, values in metrics.items() if values}

    @staticmethod
    def save_results(results: Dict[str, Any], output_path: str, format: str = "json") -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = format.lower()
        if fmt == "json":
            path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            return
        if fmt == "jsonl":
            with path.open("w", encoding="utf-8") as handle:
                for row in results.get("runs", []):
                    handle.write(json.dumps(row) + "\n")
            return
        if fmt == "csv":
            rows = results.get("runs", [])
            if not rows:
                path.write_text("", encoding="utf-8")
                return
            fieldnames = sorted({key for row in rows for key in row.keys()})
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return
        raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def generate_report(results: Dict[str, Any]) -> str:
        summary = results.get("summary", {})
        lines = ["Summary metrics:"]
        for name, value in sorted(summary.items()):
            lines.append(f"- {name}: {value:.4f}")
        return "\n".join(lines)
