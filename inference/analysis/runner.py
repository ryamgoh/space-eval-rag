from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, List, Mapping, Optional

import pandas as pd

from inference.analysis import loader, plots


@dataclass(frozen=True)
class AnalysisConfig:
    enabled: bool
    timing: str
    plot_set: str
    output_dir: Optional[str]
    image_format: str
    metrics_include: List[str]
    metrics_exclude: List[str]
    compare_runs: List[str]

    @staticmethod
    def from_config(config: Mapping[str, Any] | None) -> "AnalysisConfig":
        if not isinstance(config, Mapping):
            return AnalysisConfig(
                enabled=False,
                timing="final",
                plot_set="core",
                output_dir=None,
                image_format="png",
                metrics_include=[],
                metrics_exclude=[],
                compare_runs=[],
            )
        return AnalysisConfig(
            enabled=bool(config.get("enabled", False)),
            timing=str(config.get("timing", "final")),
            plot_set=str(config.get("plots", "core")),
            output_dir=config.get("output_dir"),
            image_format=str(config.get("image_format", "png")),
            metrics_include=list(config.get("metrics_include", []) or []),
            metrics_exclude=list(config.get("metrics_exclude", []) or []),
            compare_runs=list(config.get("compare_runs", []) or []),
        )


class AnalysisManager:
    """Orchestrate analysis plots for one or more runs."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config = AnalysisConfig.from_config(config)
        self._lock = RLock()

    def update_incremental(self, run_dir: Path) -> None:
        if not self.config.enabled:
            return
        if self.config.timing not in {"incremental", "incremental+final"}:
            return
        with self._lock:
            self._generate_run_plots(run_dir, prefer_summary=False)

    def finalize_run(self, run_dir: Path) -> None:
        if not self.config.enabled:
            return
        if self.config.timing not in {"final", "incremental", "incremental+final"}:
            return
        with self._lock:
            self._generate_run_plots(run_dir, prefer_summary=True)
            if self.config.compare_runs:
                self.compare_runs(self.config.compare_runs)

    def compare_runs(self, run_dirs: Iterable[str], out_dir: Optional[str] = None) -> None:
        with self._lock:
            self._compare_runs(run_dirs, out_dir=out_dir)

    def _compare_runs(self, run_dirs: Iterable[str], out_dir: Optional[str] = None) -> None:
        run_paths = [self._resolve_run_dir(path) for path in run_dirs]
        frames = [loader.load_run_frame(path, prefer_summary=True) for path in run_paths]
        if not frames:
            return
        df = pd.concat(frames, ignore_index=True)
        if df.empty:
            return
        metrics = self._select_metrics(df)
        output_dir = self._resolve_compare_output_dir(out_dir)
        formats = self._image_formats()
        for metric in metrics:
            plots.plot_metric_by_run(df, metric, output_dir, formats)
        plots.plot_summary_table(df, output_dir, formats)

    def _generate_run_plots(self, run_dir: Path, prefer_summary: bool) -> None:
        df = loader.load_run_frame(run_dir, prefer_summary=prefer_summary)
        if df.empty:
            return
        metrics = self._select_metrics(df)
        output_dir = self._resolve_run_output_dir(run_dir)
        formats = self._image_formats()
        for metric in metrics:
            if self.config.plot_set in {"minimal", "core", "extended"}:
                plots.plot_metric_by_model(df, metric, output_dir, formats)
            if self.config.plot_set in {"core", "extended"}:
                plots.plot_metric_by_task(df, metric, output_dir, formats)
                plots.plot_metric_overview(
                    df,
                    metric,
                    output_dir,
                    formats,
                    include_heatmap=self.config.plot_set == "extended",
                )
            if self.config.plot_set == "extended":
                plots.plot_metric_heatmap(df, metric, output_dir, formats)
        if self.config.plot_set in {"core", "extended"}:
            plots.plot_summary_table(df, output_dir, formats)

    def _select_metrics(self, df: pd.DataFrame) -> List[str]:
        metrics = sorted(df["metric_name"].dropna().unique().tolist())
        if self.config.metrics_include:
            metrics = [m for m in metrics if m in self.config.metrics_include]
        if self.config.metrics_exclude:
            metrics = [m for m in metrics if m not in self.config.metrics_exclude]
        return metrics

    def _image_formats(self) -> List[str]:
        fmt = self.config.image_format.lower()
        if fmt == "both":
            return ["png", "svg"]
        return [fmt]

    def _resolve_run_output_dir(self, run_dir: Path) -> Path:
        if self.config.output_dir:
            path = Path(self.config.output_dir)
            return path if path.is_absolute() else run_dir / path
        return run_dir / "analysis"

    def _resolve_compare_output_dir(self, out_dir: Optional[str]) -> Path:
        if out_dir:
            return Path(out_dir)
        if self.config.output_dir:
            return Path(self.config.output_dir)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return Path("runs") / "compare" / timestamp

    @staticmethod
    def _resolve_run_dir(value: str) -> Path:
        candidate = Path(value)
        if candidate.exists():
            return candidate
        fallback = Path("runs") / value
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"Run directory not found: {value}")
