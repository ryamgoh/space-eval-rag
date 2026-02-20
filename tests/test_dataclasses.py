"""Tests for dataclass behavior - ensures conversion to Pydantic preserves logic."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import pytest

from inference.adapters.base_task import ConstrainedOutputConfig, GenerationConfig
from inference.config.models import AnalysisConfig, TaskConfig


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_default_values(self):
        """Default values should match expected."""
        config = GenerationConfig()

        assert config.batch_size == 1
        assert config.batch_cb is None
        assert config.progress_cb is None
        assert config.progress_every == 1
        assert config.generation_kwargs == {}

    def test_explicit_values(self):
        """Explicit values should be set correctly."""
        callback = lambda: None
        gen_kwargs = {"max_new_tokens": 100, "temperature": 0.7}

        config = GenerationConfig(
            batch_size=4,
            batch_cb=callback,
            progress_cb=callback,
            progress_every=2,
            generation_kwargs=gen_kwargs,
        )

        assert config.batch_size == 4
        assert config.batch_cb is callback
        assert config.progress_cb is callback
        assert config.progress_every == 2
        assert config.generation_kwargs == gen_kwargs

    def test_from_kwargs_separates_control_params(self):
        """from_kwargs should separate control params from generation kwargs."""
        callback = lambda: None

        config = GenerationConfig.from_kwargs(
            batch_size=8,
            batch_cb=callback,
            progress_cb=callback,
            progress_every=5,
            max_new_tokens=128,
            temperature=0.9,
            top_p=0.95,
        )

        assert config.batch_size == 8
        assert config.batch_cb is callback
        assert config.progress_cb is callback
        assert config.progress_every == 5
        assert config.generation_kwargs == {
            "max_new_tokens": 128,
            "temperature": 0.9,
            "top_p": 0.95,
        }

    def test_from_kwargs_defaults(self):
        """from_kwargs with minimal args should use defaults."""
        config = GenerationConfig.from_kwargs(batch_size=2)

        assert config.batch_size == 2
        assert config.batch_cb is None
        assert config.progress_cb is None
        assert config.progress_every == 1
        assert config.generation_kwargs == {}

    def test_from_kwargs_with_only_generation_kwargs(self):
        """from_kwargs should handle only generation kwargs."""
        config = GenerationConfig.from_kwargs(
            batch_size=1, max_new_tokens=50, do_sample=True
        )

        assert config.batch_size == 1
        assert config.generation_kwargs == {"max_new_tokens": 50, "do_sample": True}


class TestConstrainedOutputConfig:
    """Tests for ConstrainedOutputConfig dataclass."""

    def test_default_values(self):
        """Default values should match expected."""
        config = ConstrainedOutputConfig()

        assert config.enabled is False
        assert config.choices == []
        assert config.default_choice is None
        assert config.thinking_delimiters is None

    def test_explicit_values(self):
        """Explicit values should be set correctly."""
        config = ConstrainedOutputConfig(
            enabled=True,
            choices=["A", "B", "C", "D"],
            default_choice="C",
            thinking_delimiters={"start": "<think>", "end": "</think>"},
        )

        assert config.enabled is True
        assert config.choices == ["A", "B", "C", "D"]
        assert config.default_choice == "C"
        assert config.thinking_delimiters == {"start": "<think>", "end": "</think>"}

    def test_from_task_cfg_bool_true(self):
        """from_task_cfg with bool True should enable."""
        task_cfg = TaskConfig(
            name="test",
            dataset="test",
            prompt_template="{x}",
            input_mappings={"x": "x"},
            constrained_output=True,
            choices=["yes", "no"],
        )
        config = ConstrainedOutputConfig.from_task_cfg(task_cfg)

        assert config.enabled is True
        assert config.choices == ["yes", "no"]
        assert config.default_choice is None

    def test_from_task_cfg_bool_false(self):
        """from_task_cfg with bool False should disable."""
        task_cfg = TaskConfig(
            name="test",
            dataset="test",
            prompt_template="{x}",
            input_mappings={"x": "x"},
            constrained_output=False,
            choices=["yes", "no"],
        )
        config = ConstrainedOutputConfig.from_task_cfg(task_cfg)

        assert config.enabled is False
        assert config.choices == ["yes", "no"]

    def test_from_task_cfg_nested_dict(self):
        """from_task_cfg with nested dict should parse correctly."""
        task_cfg = TaskConfig(
            name="test",
            dataset="test",
            prompt_template="{x}",
            input_mappings={"x": "x"},
            constrained_output={"enabled": True, "default_choice": "B"},
            choices=["A", "B", "C"],
        )
        config = ConstrainedOutputConfig.from_task_cfg(task_cfg)

        assert config.enabled is True
        assert config.choices == ["A", "B", "C"]
        assert config.default_choice == "B"

    def test_from_task_cfg_with_thinking_delimiters(self):
        """from_task_cfg should extract thinking_delimiters from task level."""
        task_cfg = TaskConfig(
            name="test",
            dataset="test",
            prompt_template="{x}",
            input_mappings={"x": "x"},
            constrained_output=True,
            choices=["A", "B"],
            thinking_delimiters={"start": "<think>", "end": "</think>"},
        )
        config = ConstrainedOutputConfig.from_task_cfg(task_cfg)

        assert config.thinking_delimiters == {"start": "<think>", "end": "</think>"}

    def test_from_task_cfg_missing_constrained_output(self):
        """from_task_cfg without constrained_output should default disabled."""
        task_cfg = TaskConfig(
            name="test",
            dataset="test",
            prompt_template="{x}",
            input_mappings={"x": "x"},
            choices=["A", "B"],
        )
        config = ConstrainedOutputConfig.from_task_cfg(task_cfg)

        assert config.enabled is False
        assert config.choices == ["A", "B"]

    def test_default_property_returns_default_choice(self):
        """default property should return default_choice if set."""
        config = ConstrainedOutputConfig(
            enabled=True, choices=["A", "B", "C"], default_choice="B"
        )

        assert config.default == "B"

    def test_default_property_falls_back_to_first_choice(self):
        """default property should fall back to first choice."""
        config = ConstrainedOutputConfig(enabled=True, choices=["A", "B", "C"])

        assert config.default == "A"

    def test_default_property_returns_none_if_no_choices(self):
        """default property should return None if no choices."""
        config = ConstrainedOutputConfig(enabled=True)

        assert config.default is None


class TestAnalysisConfig:
    """Tests for AnalysisConfig dataclass."""

    def test_from_config_none(self):
        """from_config with None should return defaults."""
        config = AnalysisConfig.from_config(None)

        assert config.enabled is False
        assert config.timing == "final"
        assert config.plot_set == "core"
        assert config.output_dir is None
        assert config.image_format == "png"
        assert config.metrics_include == []
        assert config.metrics_exclude == []
        assert config.compare_runs == []

    def test_from_config_empty_dict(self):
        """from_config with empty dict should return defaults."""
        config = AnalysisConfig.from_config({})

        assert config.enabled is False
        assert config.timing == "final"

    def test_from_config_with_values(self):
        """from_config should extract values correctly."""
        input_config = {
            "enabled": True,
            "timing": "incremental",
            "plots": "extended",
            "output_dir": "analysis_output",
            "image_format": "svg",
            "metrics_include": ["accuracy", "f1"],
            "metrics_exclude": ["loss"],
            "compare_runs": ["run1", "run2"],
        }
        config = AnalysisConfig.from_config(input_config)

        assert config.enabled is True
        assert config.timing == "incremental"
        assert config.plot_set == "extended"
        assert config.output_dir == "analysis_output"
        assert config.image_format == "svg"
        assert config.metrics_include == ["accuracy", "f1"]
        assert config.metrics_exclude == ["loss"]
        assert config.compare_runs == ["run1", "run2"]

    def test_from_config_uses_plots_not_plot_set(self):
        """from_config should map 'plots' to plot_set field."""
        config = AnalysisConfig.from_config({"plots": "minimal"})

        assert config.plot_set == "minimal"

    def test_frozen(self):
        """AnalysisConfig should be immutable (frozen)."""
        from pydantic import ValidationError

        config = AnalysisConfig.from_config({"enabled": True})

        with pytest.raises(ValidationError):
            config.enabled = False

    def test_metrics_include_none_converted_to_empty_list(self):
        """metrics_include with None should become empty list."""
        config = AnalysisConfig.from_config({"metrics_include": None})
        assert config.metrics_include == []

    def test_compare_runs_none_converted_to_empty_list(self):
        """compare_runs with None should become empty list."""
        config = AnalysisConfig.from_config({"compare_runs": None})
        assert config.compare_runs == []
