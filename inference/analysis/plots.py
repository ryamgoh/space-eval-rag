from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: Figure, output_base: Path, formats: Iterable[str]) -> None:
    _ensure_dir(output_base.parent)
    for fmt in formats:
        fig.savefig(
            output_base.with_suffix(f".{fmt}"),
            bbox_inches="tight",
            dpi=150,
        )


def _prep_theme() -> None:
    sns.set_theme(style="whitegrid")


def plot_metric_by_task(
    df: pd.DataFrame,
    metric_name: str,
    output_dir: Path,
    formats: List[str],
) -> None:
    data = df[df["metric_name"] == metric_name]
    if data.empty:
        return
    _prep_theme()
    tasks = data["task"].unique()
    fig, ax = plt.subplots(figsize=(max(6, len(tasks) * 0.6), 4))
    sns.barplot(data=data, x="task", y="metric_value", hue="model", ax=ax)
    ax.set_title(f"{metric_name} by task")
    ax.set_xlabel("Task")
    ax.set_ylabel(metric_name)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save_figure(fig, output_dir / "metrics" / f"{metric_name}_by_task", formats)
    plt.close(fig)


def plot_metric_by_model(
    df: pd.DataFrame,
    metric_name: str,
    output_dir: Path,
    formats: List[str],
) -> None:
    data = df[df["metric_name"] == metric_name]
    if data.empty:
        return
    grouped = (
        data.groupby("model", as_index=False)["metric_value"].mean().sort_values("model")
    )
    _prep_theme()
    fig, ax = plt.subplots(figsize=(max(6, len(grouped) * 0.6), 4))
    sns.barplot(data=grouped, x="model", y="metric_value", ax=ax)
    ax.set_title(f"{metric_name} by model (avg across tasks)")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_name)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save_figure(fig, output_dir / "metrics" / f"{metric_name}_by_model", formats)
    plt.close(fig)


def plot_metric_heatmap(
    df: pd.DataFrame,
    metric_name: str,
    output_dir: Path,
    formats: List[str],
) -> None:
    data = df[df["metric_name"] == metric_name]
    if data.empty:
        return
    pivot = data.pivot_table(
        index="task", columns="model", values="metric_value", aggfunc="mean"
    )
    if pivot.empty:
        return
    _prep_theme()
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 0.6), max(4, len(pivot) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt=".3f", linewidths=0.5, ax=ax)
    ax.set_title(f"{metric_name} heatmap")
    ax.set_xlabel("Model")
    ax.set_ylabel("Task")
    fig.tight_layout()
    _save_figure(fig, output_dir / "metrics" / f"{metric_name}_heatmap", formats)
    plt.close(fig)


def plot_summary_table(df: pd.DataFrame, output_dir: Path, formats: List[str]) -> None:
    pivot = df.pivot_table(
        index="model", columns="metric_name", values="metric_value", aggfunc="mean"
    )
    if pivot.empty:
        return
    _prep_theme()
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 0.8), max(3, len(pivot) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt=".3f", linewidths=0.5, ax=ax)
    ax.set_title("Summary metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Model")
    fig.tight_layout()
    _save_figure(fig, output_dir / "summary" / "metrics_table", formats)
    plt.close(fig)


def plot_metric_by_run(
    df: pd.DataFrame,
    metric_name: str,
    output_dir: Path,
    formats: List[str],
) -> None:
    data = df[df["metric_name"] == metric_name]
    if data.empty:
        return
    axis = "task" if data["task"].nunique() > 1 else "model"
    _prep_theme()
    categories = data[axis].unique()
    fig, ax = plt.subplots(figsize=(max(6, len(categories) * 0.6), 4))
    sns.barplot(data=data, x=axis, y="metric_value", hue="run_id", ax=ax)
    ax.set_title(f"{metric_name} by run")
    ax.set_xlabel(axis.title())
    ax.set_ylabel(metric_name)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save_figure(fig, output_dir / "compare" / f"{metric_name}_by_run", formats)
    plt.close(fig)
