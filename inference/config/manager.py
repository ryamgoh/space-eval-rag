from __future__ import annotations

import os
from typing import Any, Dict, List

import yaml


class ConfigError(ValueError):
    """Raised when the configuration file fails validation."""
    pass


class ConfigManager:
    """Loads, validates, and normalizes YAML configuration files."""
    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """Read a YAML config file, expand env vars, validate, and apply defaults."""
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = os.path.expandvars(handle.read())
        config = yaml.safe_load(raw) or {}
        ConfigManager.validate_config(config)
        return ConfigManager.merge_defaults(config)

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate the minimum required structure of the config."""
        if not isinstance(config, dict):
            raise ConfigError("Config must be a mapping.")
        if "models" not in config or not isinstance(config["models"], list):
            raise ConfigError("Config must include a list of models.")
        if "tasks" not in config or not isinstance(config["tasks"], list):
            raise ConfigError("Config must include a list of tasks.")
        if "evaluation" not in config or not isinstance(config["evaluation"], dict):
            raise ConfigError("Config must include evaluation settings.")

        for model in config["models"]:
            if "name" not in model or "type" not in model:
                raise ConfigError("Each model requires 'name' and 'type'.")
            model_type = str(model["type"]).lower()
            if model_type in {"huggingface", "vllm"} and "model_path" not in model:
                raise ConfigError(f"Model '{model.get('name', '')}' requires 'model_path'.")

        for task in config["tasks"]:
            if "name" not in task or "dataset" not in task:
                raise ConfigError("Each task requires 'name' and 'dataset'.")
            if "prompt_template" not in task:
                raise ConfigError("Each task requires prompt_template.")
            if "input_mappings" not in task:
                raise ConfigError("Each task requires input_mappings.")
            if "metrics" in task and not isinstance(task["metrics"], list):
                raise ConfigError("Task metrics must be a list.")
            rag_cfg = task.get("rag")
            if rag_cfg is None:
                if "{context}" in task.get("prompt_template", ""):
                    raise ConfigError("prompt_template includes '{context}' but rag is not configured.")
            else:
                if not isinstance(rag_cfg, dict):
                    raise ConfigError("rag must be a mapping when provided.")
                if rag_cfg.get("enabled"):
                    if "{context}" not in task.get("prompt_template", ""):
                        raise ConfigError("RAG enabled requires '{context}' in prompt_template.")
                    if "corpus" not in rag_cfg:
                        raise ConfigError("RAG enabled requires rag.corpus.")
                    if "corpus_template" not in rag_cfg:
                        raise ConfigError("RAG enabled requires rag.corpus_template.")
                    if "corpus_mappings" not in rag_cfg:
                        raise ConfigError("RAG enabled requires rag.corpus_mappings.")
                    embed_cfg = rag_cfg.get("embedding_model")
                    if not isinstance(embed_cfg, dict) or "model_path" not in embed_cfg:
                        raise ConfigError("RAG enabled requires rag.embedding_model.model_path.")
                    if "context_k" in rag_cfg:
                        context_k = rag_cfg["context_k"]
                        if not isinstance(context_k, int) or context_k < 1:
                            raise ConfigError("rag.context_k must be a positive integer.")
                    if "query_mappings" in rag_cfg and not isinstance(
                        rag_cfg["query_mappings"], dict
                    ):
                        raise ConfigError("rag.query_mappings must be a mapping.")
                    if "corpus_mappings" in rag_cfg and not isinstance(
                        rag_cfg["corpus_mappings"], dict
                    ):
                        raise ConfigError("rag.corpus_mappings must be a mapping.")
                else:
                    if "{context}" in task.get("prompt_template", ""):
                        raise ConfigError("rag.enabled is false but prompt_template includes '{context}'.")

    @staticmethod
    def merge_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values for optional config fields."""
        evaluation = config.setdefault("evaluation", {})
        evaluation.setdefault("batch_size", 8)
        evaluation.setdefault("max_concurrent", 4)
        evaluation.setdefault("output_dir", "runs")
        evaluation.setdefault("sample_size", None)
        evaluation.setdefault("prediction_sample_size", 5)
        evaluation.setdefault("save_detailed", False)
        evaluation.setdefault("max_detailed_examples", 50)
        evaluation.setdefault("checkpoint_batches", 0)
        # Optional progress logging defaults.
        evaluation.setdefault("log_progress", False)
        evaluation.setdefault("progress_every_batches", 1)
        evaluation.setdefault("log_level", "INFO")
        analysis = config.setdefault("analysis", {})
        if isinstance(analysis, dict):
            analysis.setdefault("enabled", False)
            analysis.setdefault("timing", "final")
            analysis.setdefault("plots", "core")
            analysis.setdefault("output_dir", None)
            analysis.setdefault("image_format", "png")
        config.setdefault("datasets_dir", "datasets")
        models_dir = config.setdefault("models_dir", "models")
        for model in config.get("models", []):
            if "local_dir" not in model:
                model_path = model.get("model_path") or model.get("name")
                if model_path:
                    # Preserve hub-style paths (e.g., org/model) as nested folders.
                    relative_path = model_path.strip("/").replace(" ", "_")
                    model["local_dir"] = os.path.join(models_dir, relative_path)
        for task in config.get("tasks", []):
            # Metrics are optional; default to empty for pipeline flexibility.
            task.setdefault("metrics", [])
            rag_cfg = task.get("rag")
            if isinstance(rag_cfg, dict):
                rag_cfg.setdefault("enabled", False)
                rag_cfg.setdefault("context_k", 3)
                rag_cfg.setdefault("context_template", "{text}")
                rag_cfg.setdefault("context_separator", "\n\n")
                # Keep embedding config lightweight but explicit.
                embedding_cfg = rag_cfg.setdefault("embedding_model", {})
                embedding_cfg.setdefault("pooling", "mean")
                embedding_cfg.setdefault("batch_size", 32)
                if "query_mappings" not in rag_cfg and "input_mappings" in task:
                    rag_cfg["query_mappings"] = task["input_mappings"]
                if "query_template" not in rag_cfg and "prompt_template" in task:
                    rag_cfg["query_template"] = task["prompt_template"].replace("{context}", "")
        return config
