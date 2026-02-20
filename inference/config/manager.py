"""Configuration loading and validation using Pydantic."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from inference.config.models import RootConfig


class ConfigError(ValueError):
    """Raised when the configuration file fails validation."""

    pass


def load_config(config_path: str | Path) -> RootConfig:
    """Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        RootConfig: Validated configuration object.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ConfigError: If the configuration fails validation.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        raw = os.path.expandvars(handle.read())

    data = yaml.safe_load(raw) or {}

    try:
        return RootConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigError(_format_validation_error(e)) from e


def _format_validation_error(error: ValidationError) -> str:
    """Format a Pydantic ValidationError into a readable message."""
    messages = []
    for err in error.errors():
        loc = ".".join(str(x) for x in err["loc"])
        msg = err["msg"]
        messages.append(f"{loc}: {msg}")
    return "Configuration validation failed:\n" + "\n".join(messages)
