from __future__ import annotations

import asyncio

from inference.config.models import VLLMModelConfig
from inference.model.core_model import CoreModelInterface
from inference.util.model_paths import resolve_model_path


class VLLMModel(CoreModelInterface):
    """Adapter for vLLM-backed models."""

    def __init__(self, config: VLLMModelConfig):
        """Initialize vLLM engine and sampling parameters."""
        super().__init__(
            name=config.name, model_type="vllm", max_batch_size=config.max_batch_size
        )
        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("vLLM is not available in this environment.") from exc

        resolved_path, download_dir = resolve_model_path(
            config.model_path, config.local_dir
        )
        self.sampling_params = SamplingParams(**config.sampling_kwargs)
        self.llm = LLM(model=resolved_path, download_dir=download_dir)

    async def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate responses for a batch of prompts via vLLM."""

        def _run() -> list[str]:
            outputs = self.llm.generate(prompts, self.sampling_params)
            return [out.outputs[0].text if out.outputs else "" for out in outputs]

        return await asyncio.to_thread(_run)
