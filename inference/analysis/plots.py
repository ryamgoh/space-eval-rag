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


def _place_legend_below(ax: plt.Axes, title: str | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        title=title,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(4, len(labels)),
        frameon=False,
    )


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
    _place_legend_below(ax, title="model")
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


def plot_metric_overview(
    df: pd.DataFrame,
    metric_name: str,
    output_dir: Path,
    formats: List[str],
    include_heatmap: bool = False,
) -> None:
    data = df[df["metric_name"] == metric_name]
    if data.empty:
        return
    grouped = (
        data.groupby("model", as_index=False)["metric_value"].mean().sort_values("model")
    )
    pivot = data.pivot_table(
        index="task", columns="model", values="metric_value", aggfunc="mean"
    )
    _prep_theme()
    ncols = 3 if include_heatmap else 2
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(max(10, ncols * 4.5), 4.5),
    )
    axes = list(axes) if isinstance(axes, Iterable) else [axes]

    # By task
    sns.barplot(data=data, x="task", y="metric_value", hue="model", ax=axes[0])
    axes[0].set_title("By task")
    axes[0].set_xlabel("Task")
    axes[0].set_ylabel(metric_name)
    axes[0].tick_params(axis="x", rotation=30)
    handles, labels = axes[0].get_legend_handles_labels()
    if axes[0].legend_ is not None:
        axes[0].legend_.remove()

    # By model
    sns.barplot(data=grouped, x="model", y="metric_value", ax=axes[1])
    axes[1].set_title("By model (avg)")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel(metric_name)
    axes[1].tick_params(axis="x", rotation=30)

    if include_heatmap:
        sns.heatmap(pivot, annot=True, fmt=".3f", linewidths=0.5, ax=axes[2])
        axes[2].set_title("Heatmap")
        axes[2].set_xlabel("Model")
        axes[2].set_ylabel("Task")

    if handles:
        fig.legend(
            handles,
            labels,
            title="model",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(4, len(labels)),
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save_figure(fig, output_dir / "metrics" / f"{metric_name}_overview", formats)
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
    _place_legend_below(ax, title="run_id")
    fig.tight_layout()
    _save_figure(fig, output_dir / "compare" / f"{metric_name}_by_run", formats)
    plt.close(fig)
