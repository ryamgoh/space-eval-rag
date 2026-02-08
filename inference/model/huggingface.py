from __future__ import annotations

import asyncio
from typing import Any, Dict

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
        # Device map implies HF/Accelerate will handle sharding/offload.
        self.uses_device_map = "device_map" in model_kwargs

        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_path, cache_dir=cache_dir, trust_remote_code=trust_remote, **tokenizer_kwargs
        )
        if model_class == "causal":
            self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_class = model_class
        model_cls = AutoModelForSeq2SeqLM if model_class == "seq2seq" else AutoModelForCausalLM
        self.model = model_cls.from_pretrained(
            resolved_path, cache_dir=cache_dir, trust_remote_code=trust_remote, **model_kwargs
        )
        # Default to CUDA when available, but allow config override.
        self.device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        if not self.uses_device_map:
            # Keep it simple for single-device runs; device_map users manage placement themselves.
            self.model.to(self.device)
        # Ensure inputs land on the model's expected device.
        self.input_device = self.model.device if self.uses_device_map else torch.device(self.device)
        self.model.eval()
        self.generation_kwargs = config.get("generation_kwargs", {})

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
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            if self.input_device is not None:
                inputs = inputs.to(self.input_device)
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
            # Raw keeps special tokens; clean is used for completion outputs.
            raw_full = self.tokenizer.batch_decode(sequences, skip_special_tokens=False)
            clean_full = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            if self.model_class == "causal":
                # Causal LM outputs include the prompt; keep full text and decode completion separately.
                prompt_len = inputs["input_ids"].shape[1]
                completion_seqs = sequences[:, prompt_len:]
                completion_text = self.tokenizer.batch_decode(
                    completion_seqs, skip_special_tokens=True
                )
                return completion_text, raw_full
            return clean_full, raw_full

        return await asyncio.to_thread(_run, list(prompts))

    def get_special_tokens(self) -> Dict[str, Any] | None:
        """Return tokenizer special token metadata."""
        return {
            "special_tokens_map": dict(self.tokenizer.special_tokens_map),
            "all_special_tokens": list(self.tokenizer.all_special_tokens),
            "all_special_ids": list(self.tokenizer.all_special_ids),
        }
