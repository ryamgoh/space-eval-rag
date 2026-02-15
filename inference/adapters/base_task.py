from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from datasets import Dataset

from inference.model.base_model import BaseModel


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        batch_size: Number of prompts per batch
        batch_cb: Callback for batch completion
        progress_cb: Callback for progress updates
        progress_every: Report progress every N batches
        generation_kwargs: Model-specific generation params (max_new_tokens, temperature, etc.)
    """

    batch_size: int = 1
    batch_cb: Any | None = None
    progress_cb: Any | None = None
    progress_every: int = 1
    generation_kwargs: dict | None = None

    @classmethod
    def from_kwargs(cls, batch_size: int = 1, **kwargs) -> "GenerationConfig":
        """Separate control params from generation kwargs."""
        control_keys = {"batch_cb", "progress_cb", "progress_every"}
        control_params = {k: kwargs.pop(k) for k in control_keys if k in kwargs}
        return cls(
            batch_size=batch_size,
            batch_cb=control_params.get("batch_cb"),
            progress_cb=control_params.get("progress_cb"),
            progress_every=control_params.get("progress_every", 1),
            generation_kwargs=kwargs,
        )


@dataclass
class ConstrainedOutputConfig:
    """Configuration for constrained output generation.

    Supports both simple and nested config formats:

    Simple format:
        constrained_output: true
        choices: ["A", "B", "C", "D"]

    Nested format:
        constrained_output:
          enabled: true
        choices: ["A", "B", "C", "D"]
        thinking_delimiters:
          start: "<think"
          end: "</think"

    Note:
    - thinking_delimiters should be specified at task level (not inside
      constrained_output) to allow sharing with other task features like
      strip_from_prediction.
    - Two-phase generation (for thinking models) is determined by the model's
      is_thinking_model property, not by task config.

    Attributes:
        enabled: Whether to use constrained generation
        choices: Valid choices for extraction/constraining
        default_choice: Fallback when extraction fails
        thinking_delimiters: Optional delimiters for thinking phase (from task level)
    """

    enabled: bool = False
    choices: Sequence[str] = field(default_factory=list)
    default_choice: Optional[str] = None
    thinking_delimiters: Optional[Mapping[str, str]] = None

    @classmethod
    def from_task_cfg(cls, task_cfg: Mapping[str, Any]) -> "ConstrainedOutputConfig":
        """Parse constrained output config from task config.

        thinking_delimiters is always read from task level, not from nested config.
        """
        raw = task_cfg.get("constrained_output", False)

        if isinstance(raw, bool):
            return cls(
                enabled=raw,
                choices=task_cfg.get("choices", []),
                default_choice=task_cfg.get("default_choice"),
                thinking_delimiters=task_cfg.get("thinking_delimiters"),
            )

        if isinstance(raw, Mapping):
            return cls(
                enabled=raw.get("enabled", False),
                choices=task_cfg.get("choices", []),
                default_choice=task_cfg.get("default_choice")
                or raw.get("default_choice"),
                thinking_delimiters=task_cfg.get("thinking_delimiters"),
            )

        return cls(choices=task_cfg.get("choices", []))

    @property
    def default(self) -> Optional[str]:
        """Get default choice, falling back to first choice if available."""
        if self.default_choice:
            return self.default_choice
        if self.choices:
            return list(self.choices)[0]
        return None


class BaseTaskAdapter(ABC):
    """Base interface for task adapters."""

    name = "base"

    @abstractmethod
    def build_prompts(self, dataset: Dataset, task_cfg: Mapping[str, Any]) -> List[str]:
        """Construct prompts for each dataset row."""
        raise NotImplementedError

    @abstractmethod
    def extract_references(
        self, dataset: Dataset, task_cfg: Mapping[str, Any]
    ) -> List[Any]:
        """Extract reference targets for metric computation."""
        raise NotImplementedError

    def postprocess_predictions(
        self, predictions: Sequence[str], task_cfg: Mapping[str, Any]
    ) -> List[Any]:
        """Apply task-specific postprocessing to predictions."""
        return list(predictions)

    @staticmethod
    def _build_constrained_pattern(choices: Sequence[str]) -> str:
        """Build regex pattern for constrained generation.

        For reliable constrained generation, prompts should be structured to
        guide the model to output only the choice. The pattern just constrains
        the output to valid choices.

        Args:
            choices: Valid choices (e.g., ["A", "B", "C", "D"])

        Returns:
            Regex pattern matching any of the choices

        Raises:
            ValueError: If choices is empty
        """
        if not choices:
            raise ValueError("Cannot build constrained pattern with empty choices list")
        choice_class = "|".join(re.escape(c) for c in choices)
        return rf"({choice_class})"

    async def generate_predictions(
        self,
        model: BaseModel,
        prompts: Sequence[str],
        task_cfg: Mapping[str, Any],
        batch_size: int,
        batch_cb: Any | None = None,
        **kwargs,
    ) -> Tuple[List[Any], List[Any], List[Mapping[str, Any]] | None]:
        """Generate predictions with optional constrained output.

        Supports three modes:
        1. Standard generation (constrained_output: false)
        2. Simple constrained (constrained_output: true) - for instruction-tuned models
        3. Two-phase constrained (constrained_output: true + model.is_thinking_model=True)

        The two-phase generation is automatically used when:
        - constrained_output is enabled
        - The model has is_thinking_model=True (set via config or auto-detected)

        Raises:
            ValueError: If constrained_output is enabled but choices is empty
            RuntimeError: If constrained_output is enabled but model doesn't support it
        """
        gen_config = GenerationConfig.from_kwargs(
            batch_size=batch_size, batch_cb=batch_cb, **kwargs
        )
        constrained_cfg = ConstrainedOutputConfig.from_task_cfg(task_cfg)
        task_name = task_cfg.get("name", "unknown")

        # Validate: constrained output requires non-empty choices
        if constrained_cfg.enabled and not constrained_cfg.choices:
            raise ValueError(
                f"Task '{task_name}' has constrained_output enabled "
                "but no choices specified. Please provide a non-empty 'choices' list."
            )

        # Check if model supports constrained generation
        supports_constrained = getattr(model, "supports_constrained_generation", False)

        if (
            constrained_cfg.enabled
            and constrained_cfg.choices
            and not supports_constrained
        ):
            model_type = getattr(model, "model_type", "unknown")
            hint = ""
            if model_type == "vllm":
                hint = " vLLM supports constrained generation via x-grammar, but this is not yet implemented in this framework."
            elif model_type == "api":
                hint = " API models may support constrained generation via provider-specific features, but this is not yet implemented."
            raise RuntimeError(
                f"Model '{model.name}' (type: {model_type}) does not support constrained generation. "
                f"Task '{task_name}' requires it. "
                f"Either disable constrained_output or use a HuggingFace model.{hint}"
            )

        # Two-phase generation for thinking models (model-level decision)
        is_thinking_model = getattr(model, "is_thinking_model", False)
        if constrained_cfg.enabled and is_thinking_model and constrained_cfg.choices:
            return await self._generate_two_phase(
                model, prompts, constrained_cfg, gen_config
            )

        # Simple constrained generation
        if constrained_cfg.enabled and constrained_cfg.choices:
            pattern = self._build_constrained_pattern(constrained_cfg.choices)
            predictions = await model.generate_constrained(
                list(prompts),
                pattern,
                **(gen_config.generation_kwargs or {}),
            )
            return predictions, predictions, None

        # Standard unconstrained generation
        predictions, raw_predictions = await model.batch_generate_with_prompt(
            prompts,
            batch_size=gen_config.batch_size,
            batch_cb=gen_config.batch_cb,
            progress_cb=gen_config.progress_cb,
            progress_every=gen_config.progress_every,
            **(gen_config.generation_kwargs or {}),
        )
        if raw_predictions is None:
            raw_predictions = list(predictions)
        return list(predictions), list(raw_predictions), None

    async def _generate_two_phase(
        self,
        model: BaseModel,
        prompts: Sequence[str],
        constrained_cfg: ConstrainedOutputConfig,
        gen_config: GenerationConfig,
    ) -> Tuple[List[Any], List[Any], List[Mapping[str, Any]] | None]:
        """Two-phase generation for thinking models.

        Phase 1: Generate thinking freely (model produces reasoning)
        Phase 2: Constrained generation for final answer

        Args:
            model: Model to use for generation
            prompts: Input prompts
            constrained_cfg: Constrained output configuration
            gen_config: Generation configuration

        Returns:
            Tuple of (predictions, raw_predictions, extras)
        """
        delimiters = constrained_cfg.thinking_delimiters or {}
        thinking_start = delimiters.get("start", "<think")
        thinking_end = delimiters.get("end", "</think")

        thinking_prompts = [f"{prompt}\n{thinking_start}" for prompt in prompts]

        thinking_gen_kwargs = dict(gen_config.generation_kwargs or {})
        existing_stops = thinking_gen_kwargs.get("stop_strings", [])
        if not isinstance(existing_stops, list):
            existing_stops = list(existing_stops)
        if thinking_end and thinking_end not in existing_stops:
            thinking_gen_kwargs["stop_strings"] = existing_stops + [thinking_end]

        thinking_outputs = await model.batch_generate_with_prompt(
            thinking_prompts,
            batch_size=gen_config.batch_size,
            batch_cb=gen_config.batch_cb,
            progress_cb=gen_config.progress_cb,
            progress_every=gen_config.progress_every,
            **thinking_gen_kwargs,
        )
        thinking_texts, _ = thinking_outputs

        pattern = self._build_constrained_pattern(constrained_cfg.choices)

        answer_prompts = [
            f"{prompt}\n{thinking_start}{thinking_text}\n{thinking_end}\n<answer>"
            for prompt, thinking_text in zip(prompts, thinking_texts)
        ]

        answers = await model.generate_constrained(
            answer_prompts,
            pattern,
            **(gen_config.generation_kwargs or {}),
        )

        raw_outputs = [
            f"{thinking_start}{thinking_text}\n{thinking_end}\n<answer>{answer}"
            for thinking_text, answer in zip(thinking_texts, answers)
        ]

        return answers, raw_outputs, None

    def collect_extras(
        self, task_cfg: Mapping[str, Any], count: int
    ) -> List[Mapping[str, Any]] | None:
        """Return saved per-example metadata if the adapter generates any."""
        _ = (task_cfg, count)
        return None

    def normalize_for_metrics(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
        task_cfg: Mapping[str, Any],
    ) -> Tuple[List[Any], List[Any]]:
        """Normalize predictions/references into the format metrics expect."""
        return list(predictions), list(references)
