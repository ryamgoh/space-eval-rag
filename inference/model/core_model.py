from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, Iterable, List, Optional


class CoreModelInterface(ABC):
    """Abstract base class for model adapters."""

    def __init__(
        self, name: str, model_type: str, max_batch_size: Optional[int] = None
    ):
        """Initialize basic model metadata."""
        self.name = name
        self.model_type = model_type
        self.max_batch_size = max_batch_size
        self._is_thinking_model = False

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a single completion for a prompt."""
        outputs = await self.generate_batch([prompt], **kwargs)
        return outputs[0] if outputs else ""

    async def batch_generate(
        self,
        prompts: Iterable[str],
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """Generate completions for a list of prompts, batching as needed."""
        prompts_list = list(prompts)
        if not prompts_list:
            return []
        effective_batch_size = batch_size or self.max_batch_size or len(prompts_list)
        outputs: List[str] = []
        for start in range(0, len(prompts_list), effective_batch_size):
            # Chunk to avoid blowing up model memory for large prompt lists.
            batch = prompts_list[start : start + effective_batch_size]
            outputs.extend(await self.generate_batch(batch, **kwargs))
        return outputs

    async def generate_batch_with_prompt(
        self,
        prompts: List[str],
        **kwargs,
    ) -> tuple[List[str], List[str] | None]:
        """Generate completions plus raw model outputs for a single batch."""
        outputs = await self.generate_batch(prompts, **kwargs)
        return outputs, None

    async def batch_generate_with_prompt(
        self,
        prompts: Iterable[str],
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> tuple[List[str], List[str] | None]:
        """Generate completions plus raw model outputs (if available), batching as needed."""
        # Optional callback for per-batch progress reporting.
        progress_cb = kwargs.pop("progress_cb", None)
        progress_every = kwargs.pop("progress_every", 1)
        batch_cb = kwargs.pop("batch_cb", None)
        if isinstance(progress_every, int) and progress_every < 1:
            progress_every = 1
        prompts_list = list(prompts)
        if not prompts_list:
            return [], []
        effective_batch_size = batch_size or self.max_batch_size or len(prompts_list)
        total_batches = math.ceil(len(prompts_list) / effective_batch_size)
        outputs: List[str] = []
        raw_outputs: List[str] = []
        raw_supported = True
        for batch_index, start in enumerate(
            range(0, len(prompts_list), effective_batch_size), start=1
        ):
            batch = prompts_list[start : start + effective_batch_size]
            batch_outputs, batch_raw = await self.generate_batch_with_prompt(
                batch, **kwargs
            )
            outputs.extend(batch_outputs)
            if batch_raw is None:
                raw_supported = False
            if raw_supported and batch_raw is not None:
                raw_outputs.extend(batch_raw)
            if batch_cb:
                batch_cb(
                    batch_index=batch_index,
                    total_batches=total_batches,
                    batch_start=start,
                    batch_prompts=batch,
                    batch_outputs=batch_outputs,
                    batch_raw=batch_raw,
                )
            if progress_cb and (
                batch_index == total_batches or batch_index % progress_every == 0
            ):
                progress_cb(batch_index, total_batches)
        if not raw_supported:
            return outputs, None
        return outputs, raw_outputs

    @property
    def supports_constrained_generation(self) -> bool:
        """Whether this model supports constrained output generation."""
        return False

    @property
    def is_thinking_model(self) -> bool:
        """Whether this model is a thinking/reasoning model."""
        return self._is_thinking_model

    @is_thinking_model.setter
    def is_thinking_model(self, value: bool):
        """Set whether this model is a thinking/reasoning model."""
        self._is_thinking_model = value

    async def generate_constrained(
        self,
        prompts: List[str],
        pattern: str,
        **kwargs,
    ) -> List[str]:
        """Generate with regex-constrained output.

        Override in subclasses that support constrained generation (e.g., HuggingFace with Outlines).

        Args:
            prompts: Prompts to generate from
            pattern: Regex pattern for constrained generation
            **kwargs: Additional generation kwargs

        Returns:
            List of generated outputs matching the pattern
        """
        raise NotImplementedError("Constrained generation not supported by this model")

    async def batch_generate_constrained(
        self,
        prompts: Iterable[str],
        pattern: str,
        batch_size: Optional[int] = None,
        batch_cb: Any = None,
        progress_cb: Any = None,
        progress_every: int = 1,
        **kwargs,
    ) -> tuple[List[str], List[str] | None]:
        """Generate constrained outputs with batching and callbacks.

        Args:
            prompts: Prompts to generate from
            pattern: Regex pattern for constrained generation
            batch_size: Number of prompts per batch
            batch_cb: Callback for batch completion (for checkpointing)
            progress_cb: Callback for progress updates
            progress_every: Report progress every N batches
            **kwargs: Additional generation kwargs

        Returns:
            Tuple of (outputs, raw_outputs)
        """
        if isinstance(progress_every, int) and progress_every < 1:
            progress_every = 1
        prompts_list = list(prompts)
        if not prompts_list:
            return [], []
        effective_batch_size = batch_size or self.max_batch_size or len(prompts_list)
        total_batches = math.ceil(len(prompts_list) / effective_batch_size)
        outputs: List[str] = []
        for batch_index, start in enumerate(
            range(0, len(prompts_list), effective_batch_size), start=1
        ):
            batch = prompts_list[start : start + effective_batch_size]
            batch_outputs = await self.generate_constrained(batch, pattern, **kwargs)
            outputs.extend(batch_outputs)
            if batch_cb:
                batch_cb(
                    batch_index=batch_index,
                    total_batches=total_batches,
                    batch_start=start,
                    batch_prompts=batch,
                    batch_outputs=batch_outputs,
                    batch_raw=batch_outputs,
                )
            if progress_cb and (
                batch_index == total_batches or batch_index % progress_every == 0
            ):
                progress_cb(batch_index, total_batches)
        return outputs, outputs

    def get_special_tokens(self) -> Optional[Dict[str, Any]]:
        """Return special token metadata if the backend exposes it."""
        return None

    @abstractmethod
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Backend-specific batch generation implementation."""
        raise NotImplementedError
