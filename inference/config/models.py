"""Pydantic models for configuration validation."""

from __future__ import annotations

import os
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator, field_validator


class EvaluationConfig(BaseModel):
    batch_size: int = Field(default=8, ge=1)
    max_concurrent: int = Field(default=4, ge=1)
    output_dir: str = "runs"
    sample_size: int | None = None
    prediction_sample_size: int = Field(default=5, ge=1)
    save_detailed: bool = False
    max_detailed_examples: int = Field(default=50, ge=1)
    checkpoint_batches: int = Field(default=0, ge=0)
    log_progress: bool = False
    progress_every_batches: int = Field(default=1, ge=1)
    log_level: str = "INFO"


class AnalysisConfig(BaseModel):
    enabled: bool = False
    timing: str = "final"
    plots: str = "core"
    plot_set: str = Field(default="core", validation_alias="plots")
    output_dir: str | None = None
    image_format: str = "png"
    metrics_include: list[str] = Field(default_factory=list)
    metrics_exclude: list[str] = Field(default_factory=list)
    compare_runs: list[str] = Field(default_factory=list)

    model_config = {"frozen": True, "populate_by_name": True}

    @field_validator(
        "metrics_include", "metrics_exclude", "compare_runs", mode="before"
    )
    @classmethod
    def none_to_empty_list(cls, v):
        if v is None:
            return []
        return v

    @staticmethod
    def from_config(config: dict[str, Any] | None) -> "AnalysisConfig":
        if config is None:
            return AnalysisConfig()
        return AnalysisConfig.model_validate(config)


class EmbeddingModelConfig(BaseModel):
    model_path: str
    local_dir: str | None = None
    pooling: str = "mean"
    batch_size: int = Field(default=32, ge=1)
    max_length: int = Field(default=256, ge=1)
    device: str | None = None
    trust_remote_code: bool = False


class RAGConfig(BaseModel):
    enabled: bool = False
    corpus: dict[str, Any] | None = None
    corpus_split: str | None = None
    corpus_template: str | None = None
    corpus_mappings: dict[str, str] | None = None
    context_k: int = Field(default=3, ge=1)
    context_template: str = "{text}"
    context_separator: str = "\n\n"
    embedding_model: EmbeddingModelConfig | None = None
    query_mappings: dict[str, str] | None = None
    query_template: str | None = None

    @model_validator(mode="after")
    def validate_rag_enabled(self) -> "RAGConfig":
        if not self.enabled:
            return self

        errors = []
        if self.corpus is None:
            errors.append("corpus is required when RAG is enabled")
        if not self.corpus_template:
            errors.append("corpus_template is required when RAG is enabled")
        if not self.corpus_mappings:
            errors.append("corpus_mappings is required when RAG is enabled")
        if self.embedding_model is None or not self.embedding_model.model_path:
            errors.append("embedding_model.model_path is required when RAG is enabled")

        if errors:
            raise ValueError("RAG validation errors: " + "; ".join(errors))
        return self


class HuggingFaceModelConfig(BaseModel):
    type: Literal["huggingface"]
    name: str
    model_path: str
    local_dir: str | None = None
    model_class: str = "causal"
    max_batch_size: int | None = None
    device: str | None = None
    trust_remote_code: bool = False
    tokenizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    generation_kwargs: dict[str, Any] = Field(default_factory=dict)
    sampling_kwargs: dict[str, Any] = Field(default_factory=dict)
    is_thinking_model: bool | None = None

    @model_validator(mode="after")
    def set_local_dir(self) -> "HuggingFaceModelConfig":
        if self.local_dir is None:
            relative_path = self.model_path.strip("/").replace(" ", "_")
            self.local_dir = os.path.join("models", relative_path)
        return self


class VLLMModelConfig(BaseModel):
    type: Literal["vllm"]
    name: str
    model_path: str
    local_dir: str | None = None
    max_batch_size: int | None = None
    generation_kwargs: dict[str, Any] = Field(default_factory=dict)
    sampling_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_local_dir(self) -> "VLLMModelConfig":
        if self.local_dir is None:
            relative_path = self.model_path.strip("/").replace(" ", "_")
            self.local_dir = os.path.join("models", relative_path)
        return self


class APIModelConfig(BaseModel):
    type: Literal["api"]
    name: str
    endpoint: str | None = None
    local_dir: str | None = None
    max_batch_size: int | None = None
    generation_kwargs: dict[str, Any] = Field(default_factory=dict)
    sampling_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_local_dir(self) -> "APIModelConfig":
        if self.local_dir is None:
            self.local_dir = os.path.join("models", self.name)
        return self


ModelConfig = Annotated[
    HuggingFaceModelConfig | VLLMModelConfig | APIModelConfig,
    Field(discriminator="type"),
]


class MetricConfig(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    adapter: str = "hf"


class TaskConfig(BaseModel):
    name: str
    task_adapter: str | None = None
    dataset: str | dict[str, Any]
    prompt_template: str
    input_mappings: dict[str, str]
    reference_field: str | None = None
    split: str | None = None
    sample_size: int | None = None
    metrics: list[MetricConfig | str] = Field(default_factory=list)
    label_map: list[str] | None = None
    generation_kwargs: dict[str, Any] = Field(default_factory=dict)
    rag: RAGConfig | None = None
    constrained_output: bool | dict[str, Any] = False
    choices: list[str] = Field(default_factory=list)
    default_choice: str | None = None
    thinking_delimiters: dict[str, str] | None = None
    strip_from_prediction: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_context_placeholder(self) -> "TaskConfig":
        has_context = "{context}" in self.prompt_template
        rag_enabled = self.rag.enabled if self.rag else False

        if has_context and self.rag is None:
            raise ValueError(
                f"Task '{self.name}': prompt_template includes '{{context}}' but rag is not configured."
            )

        if self.rag is not None:
            if has_context and not rag_enabled:
                raise ValueError(
                    f"Task '{self.name}': rag.enabled is false but prompt_template includes '{{context}}'."
                )
            if not has_context and rag_enabled:
                raise ValueError(
                    f"Task '{self.name}': RAG enabled requires '{{context}}' in prompt_template."
                )

        if self.rag is not None and self.rag.enabled:
            if self.rag.query_mappings is None:
                self.rag.query_mappings = dict(self.input_mappings)
            if self.rag.query_template is None:
                self.rag.query_template = self.prompt_template.replace("{context}", "")

        return self


class RootConfig(BaseModel):
    models: list[ModelConfig]
    tasks: list[TaskConfig]
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    datasets_dir: str = "datasets"
    models_dir: str = "models"
