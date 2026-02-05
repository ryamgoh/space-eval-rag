from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional


class BaseModel(ABC):
    """Abstract base class for model adapters."""
    def __init__(self, name: str, model_type: str, max_batch_size: Optional[int] = None):
        """Initialize basic model metadata."""
        self.name = name
        self.model_type = model_type
        self.max_batch_size = max_batch_size

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
        prompts_list = list(prompts)
        if not prompts_list:
            return [], []
        effective_batch_size = batch_size or self.max_batch_size or len(prompts_list)
        outputs: List[str] = []
        raw_outputs: List[str] = []
        raw_supported = True
        for start in range(0, len(prompts_list), effective_batch_size):
            batch = prompts_list[start : start + effective_batch_size]
            batch_outputs, batch_raw = await self.generate_batch_with_prompt(batch, **kwargs)
            outputs.extend(batch_outputs)
            if batch_raw is None:
                raw_supported = False
            if raw_supported:
                raw_outputs.extend(batch_raw)
        if not raw_supported:
            return outputs, None
        return outputs, raw_outputs

    @abstractmethod
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Backend-specific batch generation implementation."""
        raise NotImplementedError
