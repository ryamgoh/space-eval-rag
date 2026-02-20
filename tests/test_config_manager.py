"""Tests for Pydantic config models - ensures validation logic is preserved."""

import os
from pathlib import Path

import pytest
import yaml

from inference.config.manager import ConfigError, load_config


def write_yaml(content: str, tmp_path: Path) -> str:
    """Helper to write YAML content to a temp file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content)
    return str(config_path)


class TestConfigLoading:
    """Tests for YAML loading and parsing."""

    def test_load_valid_config(self, tmp_path: Path):
        """A valid config should load without errors."""
        config_content = """
models:
  - name: "test-model"
    type: "huggingface"
    model_path: "test/model"

tasks:
  - name: "test-task"
    dataset: "test-dataset"
    prompt_template: "Test: {input}"
    input_mappings:
      input: "text"

evaluation:
  batch_size: 4
"""
        config_path = write_yaml(config_content, tmp_path)
        result = load_config(config_path)

        assert len(result.models) == 1
        assert len(result.tasks) == 1
        assert result.models[0].name == "test-model"

    def test_load_empty_file_raises_error(self, tmp_path: Path):
        """An empty YAML file should raise ConfigError."""
        config_path = write_yaml("", tmp_path)
        with pytest.raises(ConfigError, match="models"):
            load_config(config_path)

    def test_load_nonexistent_file_raises_filenotfound(self, tmp_path: Path):
        """Loading a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_env_var_expansion(self, tmp_path: Path, monkeypatch):
        """Environment variables in YAML should be expanded."""
        monkeypatch.setenv("TEST_MODEL_PATH", "models/test-model")
        config_content = """
models:
  - name: "test"
    type: "huggingface"
    model_path: "$TEST_MODEL_PATH"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{input}"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        result = load_config(config_path)
        assert result.models[0].model_path == "models/test-model"


class TestConfigValidationModels:
    """Tests for model configuration validation."""

    def test_model_missing_name_raises_error(self, tmp_path: Path):
        """Model without 'name' should raise ConfigError."""
        config_content = """
models:
  - type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{input}"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="name"):
            load_config(config_path)

    def test_model_missing_type_raises_error(self, tmp_path: Path):
        """Model without 'type' should raise ConfigError."""
        config_content = """
models:
  - name: "test-model"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{input}"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="type"):
            load_config(config_path)

    def test_huggingface_model_missing_path_raises_error(self, tmp_path: Path):
        """HuggingFace model without model_path should raise ConfigError."""
        config_content = """
models:
  - name: "test"
    type: "huggingface"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{input}"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="model_path"):
            load_config(config_path)

    def test_vllm_model_missing_path_raises_error(self, tmp_path: Path):
        """VLLM model without model_path should raise ConfigError."""
        config_content = """
models:
  - name: "test"
    type: "vllm"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{input}"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="model_path"):
            load_config(config_path)

    def test_api_model_does_not_require_model_path(self, tmp_path: Path):
        """API model type should NOT require model_path."""
        config_content = """
models:
  - name: "test-api"
    type: "api"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{input}"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        result = load_config(config_path)
        assert result.models[0].type == "api"

    def test_invalid_model_type_raises_error(self, tmp_path: Path):
        """Invalid model type should raise ConfigError."""
        config_content = """
models:
  - name: "test"
    type: "invalid_type"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{input}"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError):
            load_config(config_path)


class TestConfigValidationTasks:
    """Tests for task configuration validation."""

    def test_task_missing_name_raises_error(self, tmp_path: Path):
        """Task without 'name' should raise ConfigError."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - dataset: "d"
    prompt_template: "{input}"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="name"):
            load_config(config_path)

    def test_task_missing_dataset_raises_error(self, tmp_path: Path):
        """Task without 'dataset' should raise ConfigError."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    prompt_template: "{input}"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="dataset"):
            load_config(config_path)

    def test_task_missing_prompt_template_raises_error(self, tmp_path: Path):
        """Task without 'prompt_template' should raise ConfigError."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="prompt_template"):
            load_config(config_path)

    def test_task_missing_input_mappings_raises_error(self, tmp_path: Path):
        """Task without 'input_mappings' should raise ConfigError."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{input}"
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="input_mappings"):
            load_config(config_path)

    def test_task_metrics_not_list_raises_error(self, tmp_path: Path):
        """Task metrics must be a list if provided."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{input}"
    input_mappings:
      input: "text"
    metrics: "rouge"
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="metrics"):
            load_config(config_path)


class TestConfigValidationRAG:
    """Tests for RAG configuration validation."""

    def test_rag_enabled_requires_context_placeholder(self, tmp_path: Path):
        """RAG enabled requires {context} in prompt_template."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "No context here"
    input_mappings:
      input: "text"
    rag:
      enabled: true
      corpus: {}
      corpus_template: "{text}"
      corpus_mappings:
        text: "content"
      embedding_model:
        model_path: "embeddings"
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="context"):
            load_config(config_path)

    def test_rag_enabled_missing_corpus_raises_error(self, tmp_path: Path):
        """RAG enabled requires corpus configuration."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{context} Answer:"
    input_mappings:
      input: "text"
    rag:
      enabled: true
      corpus_template: "{text}"
      corpus_mappings:
        text: "content"
      embedding_model:
        model_path: "embeddings"
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="corpus"):
            load_config(config_path)

    def test_rag_enabled_missing_corpus_template_raises_error(self, tmp_path: Path):
        """RAG enabled requires corpus_template."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{context} Answer:"
    input_mappings:
      input: "text"
    rag:
      enabled: true
      corpus: {}
      corpus_mappings:
        text: "content"
      embedding_model:
        model_path: "embeddings"
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="corpus_template"):
            load_config(config_path)

    def test_rag_enabled_missing_corpus_mappings_raises_error(self, tmp_path: Path):
        """RAG enabled requires corpus_mappings."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{context} Answer:"
    input_mappings:
      input: "text"
    rag:
      enabled: true
      corpus: {}
      corpus_template: "{text}"
      embedding_model:
        model_path: "embeddings"
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="corpus_mappings"):
            load_config(config_path)

    def test_rag_enabled_missing_embedding_model_path_raises_error(
        self, tmp_path: Path
    ):
        """RAG enabled requires embedding_model.model_path."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{context} Answer:"
    input_mappings:
      input: "text"
    rag:
      enabled: true
      corpus: {}
      corpus_template: "{text}"
      corpus_mappings:
        text: "content"
      embedding_model: {}
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="model_path"):
            load_config(config_path)

    def test_rag_context_placeholder_without_rag_raises_error(self, tmp_path: Path):
        """{context} in template without RAG config raises error."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{context} Answer:"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="rag is not configured"):
            load_config(config_path)

    def test_rag_disabled_with_context_placeholder_raises_error(self, tmp_path: Path):
        """RAG disabled with {context} in template raises error."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{context} Answer:"
    input_mappings:
      input: "text"
    rag:
      enabled: false
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="rag.enabled is false"):
            load_config(config_path)

    def test_rag_context_k_must_be_positive_integer(self, tmp_path: Path):
        """RAG context_k must be a positive integer."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{context} Answer:"
    input_mappings:
      input: "text"
    rag:
      enabled: true
      corpus: {}
      corpus_template: "{text}"
      corpus_mappings:
        text: "content"
      embedding_model:
        model_path: "embeddings"
      context_k: 0
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="context_k"):
            load_config(config_path)

    def test_rag_query_mappings_must_be_mapping(self, tmp_path: Path):
        """RAG query_mappings must be a dict if provided."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{context} Answer:"
    input_mappings:
      input: "text"
    rag:
      enabled: true
      corpus: {}
      corpus_template: "{text}"
      corpus_mappings:
        text: "content"
      embedding_model:
        model_path: "embeddings"
      query_mappings: "invalid"
"""
        config_path = write_yaml(config_content, tmp_path)
        with pytest.raises(ConfigError, match="query_mappings"):
            load_config(config_path)

    def test_valid_rag_config_loads_successfully(self, tmp_path: Path):
        """A complete valid RAG config should load without errors."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{context} Answer: {input}"
    input_mappings:
      input: "text"
    rag:
      enabled: true
      corpus:
        path: "corpus"
      corpus_template: "{text}"
      corpus_mappings:
        text: "content"
      embedding_model:
        model_path: "embeddings"
"""
        config_path = write_yaml(config_content, tmp_path)
        result = load_config(config_path)
        assert result.tasks[0].rag.enabled is True


class TestConfigDefaults:
    """Tests for default value application."""

    def test_evaluation_defaults_applied(self, tmp_path: Path):
        """Missing evaluation fields should get defaults."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{input}"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        result = load_config(config_path)
        eval_config = result.evaluation

        assert eval_config.batch_size == 8
        assert eval_config.max_concurrent == 4
        assert eval_config.output_dir == "runs"
        assert eval_config.sample_size is None
        assert eval_config.prediction_sample_size == 5
        assert eval_config.save_detailed is False
        assert eval_config.max_detailed_examples == 50
        assert eval_config.checkpoint_batches == 0
        assert eval_config.log_progress is False
        assert eval_config.progress_every_batches == 1
        assert eval_config.log_level == "INFO"

    def test_explicit_evaluation_values_override_defaults(self, tmp_path: Path):
        """Explicit values should override defaults."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{input}"
    input_mappings:
      input: "text"

evaluation:
  batch_size: 16
  max_concurrent: 8
"""
        config_path = write_yaml(config_content, tmp_path)
        result = load_config(config_path)

        assert result.evaluation.batch_size == 16
        assert result.evaluation.max_concurrent == 8

    def test_analysis_defaults_applied(self, tmp_path: Path):
        """Missing analysis fields should get defaults."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{input}"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        result = load_config(config_path)
        analysis = result.analysis

        assert analysis.enabled is False
        assert analysis.timing == "final"
        assert analysis.plots == "core"
        assert analysis.output_dir is None
        assert analysis.image_format == "png"

    def test_top_level_defaults_applied(self, tmp_path: Path):
        """Top-level directories should get defaults."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{input}"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        result = load_config(config_path)

        assert result.datasets_dir == "datasets"
        assert result.models_dir == "models"

    def test_model_local_dir_default_from_model_path(self, tmp_path: Path):
        """Model local_dir should default based on model_path."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "org/model-name"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{input}"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        result = load_config(config_path)

        assert result.models[0].local_dir == "models/org/model-name"

    def test_model_local_dir_default_from_name_if_no_path(self, tmp_path: Path):
        """Model local_dir should default from name if no model_path."""
        config_content = """
models:
  - name: "my-model"
    type: "api"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{input}"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        result = load_config(config_path)

        assert result.models[0].local_dir == "models/my-model"

    def test_task_metrics_defaults_to_empty_list(self, tmp_path: Path):
        """Task metrics should default to empty list."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{input}"
    input_mappings:
      input: "text"
"""
        config_path = write_yaml(config_content, tmp_path)
        result = load_config(config_path)

        assert result.tasks[0].metrics == []

    def test_rag_defaults_applied(self, tmp_path: Path):
        """RAG fields should get defaults."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{context} Answer: {input}"
    input_mappings:
      input: "text"
    rag:
      enabled: true
      corpus: {}
      corpus_template: "{text}"
      corpus_mappings:
        text: "content"
      embedding_model:
        model_path: "embeddings"
"""
        config_path = write_yaml(config_content, tmp_path)
        result = load_config(config_path)
        rag = result.tasks[0].rag

        assert rag.context_k == 3
        assert rag.context_template == "{text}"
        assert rag.context_separator == "\n\n"
        assert rag.embedding_model.pooling == "mean"
        assert rag.embedding_model.batch_size == 32

    def test_rag_query_mappings_defaults_to_input_mappings(self, tmp_path: Path):
        """RAG query_mappings should default to task input_mappings."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{context} Answer: {input}"
    input_mappings:
      input: "text"
      other: "field"
    rag:
      enabled: true
      corpus: {}
      corpus_template: "{text}"
      corpus_mappings:
        text: "content"
      embedding_model:
        model_path: "embeddings"
"""
        config_path = write_yaml(config_content, tmp_path)
        result = load_config(config_path)
        rag = result.tasks[0].rag

        assert rag.query_mappings == {"input": "text", "other": "field"}

    def test_rag_query_template_defaults_from_prompt_without_context(
        self, tmp_path: Path
    ):
        """RAG query_template should default to prompt_template without {context}."""
        config_content = """
models:
  - name: "m"
    type: "huggingface"
    model_path: "test"

tasks:
  - name: "t"
    dataset: "d"
    prompt_template: "{context} Answer the question: {input}"
    input_mappings:
      input: "text"
    rag:
      enabled: true
      corpus: {}
      corpus_template: "{text}"
      corpus_mappings:
        text: "content"
      embedding_model:
        model_path: "embeddings"
"""
        config_path = write_yaml(config_content, tmp_path)
        result = load_config(config_path)
        rag = result.tasks[0].rag

        assert rag.query_template == " Answer the question: {input}"


class TestSampleConfigs:
    """Tests using the actual sample config files."""

    def test_sample_config_loads(self):
        """The sample_config.yaml should load without errors."""
        config_path = (
            Path(__file__).parent.parent / "inference" / "config" / "sample_config.yaml"
        )
        result = load_config(str(config_path))

        assert len(result.models) >= 1
        assert len(result.tasks) >= 1
        assert result.evaluation is not None

    def test_tiny_test_config_loads(self):
        """The tiny_test_config.yaml should load without errors."""
        config_path = (
            Path(__file__).parent.parent
            / "inference"
            / "config"
            / "tiny_test_config.yaml"
        )
        result = load_config(str(config_path))

        assert len(result.models) == 3
        assert len(result.tasks) == 2
        assert result.evaluation.batch_size == 2
