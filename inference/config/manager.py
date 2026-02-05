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

        for task in config["tasks"]:
            if "name" not in task or "dataset" not in task:
                raise ConfigError("Each task requires 'name' and 'dataset'.")
            if "prompt_template" not in task and "prompt_templates" not in task:
                raise ConfigError("Each task requires prompt_template or prompt_templates.")
            if "input_mappings" not in task and "fields" not in task:
                raise ConfigError("Each task requires input_mappings or fields.")
            if "metrics" in task and not isinstance(task["metrics"], list):
                raise ConfigError("Task metrics must be a list.")

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
        config.setdefault("datasets_dir", "datasets")
        models_dir = config.setdefault("models_dir", "models")
        for model in config.get("models", []):
            if "local_dir" not in model:
                model_path = model.get("model_path") or model.get("name")
                if model_path:
                    safe_name = model_path.replace("/", "_").replace(" ", "_")
                    model["local_dir"] = os.path.join(models_dir, safe_name)
        for task in config.get("tasks", []):
            # Metrics are optional; default to empty for pipeline flexibility.
            task.setdefault("metrics", [])
        return config
