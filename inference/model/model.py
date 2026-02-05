from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from inference.model.base_model import BaseModel


def _resolve_model_path(model_path: str, local_dir: Optional[str]) -> tuple[str, Optional[str]]:
    """Decide whether to load from local dir or download into it."""
    if not local_dir:
        return model_path, None

    os.makedirs(local_dir, exist_ok=True)
    marker_files = ("config.json", "model.safetensors", "pytorch_model.bin")
    # Treat local_dir as a model path only if it already looks populated.
    has_model_files = any(os.path.exists(os.path.join(local_dir, name)) for name in marker_files)
    if has_model_files:
        return local_dir, local_dir
    return model_path, local_dir


class HuggingFaceModel(BaseModel):
    """Adapter for Hugging Face Transformers models."""
    def __init__(self, config: Dict[str, Any]):
        """Initialize tokenizer/model and cache configuration."""
        name = config["name"]
        super().__init__(name=name, model_type="huggingface", max_batch_size=config.get("max_batch_size"))
        model_path = config["model_path"]
        local_dir = config.get("local_dir")
        resolved_path, cache_dir = _resolve_model_path(model_path, local_dir)

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
        gen_kwargs = {**self.generation_kwargs, **kwargs}

        def _run() -> list[str]:
            # Run generation in a worker thread to avoid blocking the event loop.
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            if self.input_device is not None:
                inputs = inputs.to(self.input_device)
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return await asyncio.to_thread(_run)


class VLLMModel(BaseModel):
    """Adapter for vLLM-backed models."""
    def __init__(self, config: Dict[str, Any]):
        """Initialize vLLM engine and sampling parameters."""
        name = config["name"]
        super().__init__(name=name, model_type="vllm", max_batch_size=config.get("max_batch_size"))
        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("vLLM is not available in this environment.") from exc

        model_path = config["model_path"]
        local_dir = config.get("local_dir")
        resolved_path, download_dir = _resolve_model_path(model_path, local_dir)
        self.sampling_params = SamplingParams(**config.get("sampling_kwargs", {}))
        self.llm = LLM(model=resolved_path, download_dir=download_dir)

    async def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate responses for a batch of prompts via vLLM."""
        def _run() -> list[str]:
            outputs = self.llm.generate(prompts, self.sampling_params)
            return [out.outputs[0].text if out.outputs else "" for out in outputs]

        return await asyncio.to_thread(_run)


class APIModel(BaseModel):
    """Adapter placeholder for API-backed models."""
    def __init__(self, config: Dict[str, Any]):
        """Store API configuration without initializing a client."""
        name = config["name"]
        super().__init__(name=name, model_type="api", max_batch_size=config.get("max_batch_size"))
        self.config = config

    async def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate responses via an external API (not implemented)."""
        raise NotImplementedError("API model adapter is not implemented yet.")


class ModelFactory:
    """Factory for creating model adapters from config."""
    @staticmethod
    def get_model(model_config: Dict[str, Any]) -> BaseModel:
        """Instantiate the appropriate model adapter based on config type."""
        model_type = model_config["type"].lower()
        if model_type == "huggingface":
            return HuggingFaceModel(model_config)
        if model_type == "vllm":
            return VLLMModel(model_config)
        if model_type == "api":
            return APIModel(model_config)
        raise ValueError(f"Unknown model type: {model_type}")
