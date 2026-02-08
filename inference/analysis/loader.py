from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_task_results(run_dir: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for model_dir in run_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for result_path in model_dir.glob("*.json"):
            if result_path.name == "summary.json":
                continue
            payload = _read_json(result_path)
            runs = payload.get("runs", [])
            if isinstance(runs, list):
                results.extend([r for r in runs if isinstance(r, dict)])
    return results


def load_run_results(run_dir: Path, prefer_summary: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
    run_dir = run_dir.resolve()
    run_id = run_dir.name
    summary_path = run_dir / "summary.json"
    if prefer_summary and summary_path.exists():
        payload = _read_json(summary_path)
        runs = payload.get("runs", [])
        if isinstance(runs, list):
            return run_id, [r for r in runs if isinstance(r, dict)]
    return run_id, _collect_task_results(run_dir)


def results_to_frame(run_id: str, results: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in results:
        metrics = result.get("metrics", {}) or {}
        for name, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "model": result.get("model"),
                    "task": result.get("task"),
                    "metric_name": name,
                    "metric_value": float(value),
                    "num_examples": result.get("num_examples"),
                }
            )
    return pd.DataFrame(rows)


def load_run_frame(run_dir: Path, prefer_summary: bool = True) -> pd.DataFrame:
    run_id, results = load_run_results(run_dir, prefer_summary=prefer_summary)
    return results_to_frame(run_id, results)
