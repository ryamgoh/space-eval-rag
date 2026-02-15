from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from inference.model.base_model import BaseModel
from inference.util.model_paths import resolve_model_path


class HuggingFaceModel(BaseModel):
    """Adapter for Hugging Face Transformers models."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize tokenizer/model and cache configuration."""
        name = config["name"]
        super().__init__(
            name=name,
            model_type="huggingface",
            max_batch_size=config.get("max_batch_size"),
        )
        model_path = config["model_path"]
        local_dir = config.get("local_dir")
        resolved_path, cache_dir = resolve_model_path(model_path, local_dir)

        tokenizer_kwargs = config.get("tokenizer_kwargs", {})
        model_kwargs = config.get("model_kwargs", {})
        model_class = config.get("model_class", "causal")
        trust_remote = config.get("trust_remote_code", False)
        self.uses_device_map = "device_map" in model_kwargs

        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote,
            **tokenizer_kwargs,
        )
        if model_class == "causal":
            self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_class = model_class
        model_cls = (
            AutoModelForSeq2SeqLM if model_class == "seq2seq" else AutoModelForCausalLM
        )
        self.model = model_cls.from_pretrained(
            resolved_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote,
            **model_kwargs,
        )
        self.device = config.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if not self.uses_device_map:
            self.model.to(self.device)
        self.input_device = (
            self.model.device if self.uses_device_map else torch.device(self.device)
        )
        self.model.eval()
        self.generation_kwargs = config.get("generation_kwargs", {})
        self._outlines_model = None
        self._outlines_generator = None

        # Set thinking model flag from config or auto-detect from model path
        is_thinking = config.get("is_thinking_model", None)
        if is_thinking is None:
            # Auto-detect from model path
            model_path_lower = model_path.lower()
            is_thinking = (
                "thinking" in model_path_lower or "reasoning" in model_path_lower
            )
        self._is_thinking_model = bool(is_thinking)

    @property
    def supports_constrained_generation(self) -> bool:
        """HuggingFace models support constrained generation via Outlines."""
        return True

    def _init_outlines(self):
        """Lazily initialize Outlines model wrapper."""
        if self._outlines_model is not None:
            return
        import outlines

        self._outlines_model = outlines.from_transformers(self.model, self.tokenizer)

    def _get_outlines_generator(self, pattern: str):
        """Get or create an Outlines generator for the given pattern."""
        if self._outlines_generator is None or self._outlines_generator[0] != pattern:
            import outlines

            self._init_outlines()
            generator = outlines.Generator(
                self._outlines_model, output_type=outlines.regex(pattern)
            )
            self._outlines_generator = (pattern, generator)
        return self._outlines_generator[1]

    async def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate responses for a batch of prompts."""
        outputs, _ = await self.generate_batch_with_prompt(prompts, **kwargs)
        return outputs

    async def generate_batch_with_prompt(
        self,
        prompts: list[str],
        **kwargs,
    ) -> tuple[list[str], list[str] | None]:
        """Generate completions plus raw model outputs (prompt + completion for causal)."""
        gen_kwargs = {**self.generation_kwargs, **kwargs}

        def _run(batch: list[str]) -> tuple[list[str], list[str]]:
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            )
            if self.input_device is not None:
                inputs = inputs.to(self.input_device)
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
            raw_full = self.tokenizer.batch_decode(sequences, skip_special_tokens=False)
            clean_full = self.tokenizer.batch_decode(
                sequences, skip_special_tokens=True
            )
            if self.model_class == "causal":
                prompt_len = inputs["input_ids"].shape[1]
                completion_seqs = sequences[:, prompt_len:]
                completion_text = self.tokenizer.batch_decode(
                    completion_seqs, skip_special_tokens=True
                )
                return completion_text, raw_full
            return clean_full, raw_full

        return await asyncio.to_thread(_run, list(prompts))

    async def generate_constrained(
        self,
        prompts: List[str],
        pattern: str,
        **kwargs,
    ) -> List[str]:
        """Generate with regex-constrained output using Outlines.

        Args:
            prompts: Prompts to generate from
            pattern: Regex pattern for constrained generation
            **kwargs: Generation kwargs (max_new_tokens, temperature, etc.)

        Returns:
            List of generated outputs matching the pattern
        """
        gen_kwargs = {**self.generation_kwargs, **kwargs}

        def _run(batch: list[str]) -> list[str]:
            generator = self._get_outlines_generator(pattern)
            return [generator(prompt, **gen_kwargs) for prompt in batch]

        return await asyncio.to_thread(_run, list(prompts))

    def get_special_tokens(self) -> Dict[str, Any] | None:
        """Return tokenizer special token metadata."""
        return {
            "special_tokens_map": dict(self.tokenizer.special_tokens_map),
            "all_special_tokens": list(self.tokenizer.all_special_tokens),
            "all_special_ids": list(self.tokenizer.all_special_ids),
        }
