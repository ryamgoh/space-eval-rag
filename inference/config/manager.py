from __future__ import annotations

import os
from typing import Any, Dict, List

import yaml


class ConfigError(ValueError):
    pass


class ConfigManager:
    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = os.path.expandvars(handle.read())
        config = yaml.safe_load(raw) or {}
        ConfigManager.validate_config(config)
        return ConfigManager.merge_defaults(config)

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
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
            if "prompt_template" not in task:
                raise ConfigError("Each task requires a prompt_template.")
            if "input_mappings" not in task:
                raise ConfigError("Each task requires input_mappings.")
            if "metrics" not in task:
                raise ConfigError("Each task requires metrics.")

    @staticmethod
    def merge_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        evaluation = config.setdefault("evaluation", {})
        evaluation.setdefault("batch_size", 8)
        evaluation.setdefault("max_concurrent", 4)
        evaluation.setdefault("output_dir", "runs")
        evaluation.setdefault("sample_size", None)
        evaluation.setdefault("prediction_sample_size", 5)
        return config
