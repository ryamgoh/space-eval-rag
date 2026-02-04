from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from datasets import Dataset

from inference.adapters.base_task import BaseTaskAdapter
from inference.model.model import HuggingFaceModel
from inference.task.processor import TaskProcessor


class LikertLogProbTaskAdapter(BaseTaskAdapter):
    """Likert task adapter that ranks ratings using logprob scoring."""
    name = "likert_logprob"

    def build_prompts(self, dataset: Dataset, task_cfg: Mapping[str, Any]) -> List[str]:
        """Render prompts with the provided template and mappings."""
        return TaskProcessor.apply_template(
            dataset, task_cfg["prompt_template"], task_cfg["input_mappings"]
        )

    def extract_references(self, dataset: Dataset, task_cfg: Mapping[str, Any]) -> List[Any]:
        """Extract reference values for each example."""
        return TaskProcessor.extract_references(dataset, task_cfg)

    async def generate_predictions(
        self,
        model,
        prompts: Sequence[str],
        task_cfg: Mapping[str, Any],
        batch_size: int,
        **kwargs,
    ) -> Tuple[List[Any], List[Mapping[str, Any]] | None]:
        """Score each rating via loss and select the best choice."""
        if not isinstance(model, HuggingFaceModel):
            raise ValueError("likert_logprob requires an open-weight HuggingFace model.")
        scale = task_cfg.get("scale")
        if not scale:
            raise ValueError("likert_logprob requires 'scale'.")
        choices = [str(item) for item in scale]
        normalize = task_cfg.get("logprob_normalize", "mean").lower()
        if normalize not in ("mean", "sum"):
            raise ValueError("logprob_normalize must be 'mean' or 'sum'.")

        def _score_prompt(prompt: str) -> dict[str, float]:
            scores: dict[str, float] = {}
            for choice in choices:
                # Negative cross-entropy is used as a proxy logprob score.
                if model.model_class == "seq2seq":
                    inputs = model.tokenizer(prompt, return_tensors="pt")
                    labels = model.tokenizer(choice, return_tensors="pt").input_ids
                    inputs = {key: value.to(model.model.device) for key, value in inputs.items()}
                    labels = labels.to(model.model.device)
                    labels_mask = labels != model.tokenizer.pad_token_id
                    token_count = int(labels_mask.sum().item()) or 1
                    labels = labels.masked_fill(~labels_mask, -100)
                    with torch.no_grad():
                        outputs = model.model(**inputs, labels=labels)
                    loss = float(outputs.loss.item())
                    score = -loss if normalize == "mean" else -loss * token_count
                else:
                    prompt_ids = model.tokenizer(prompt, return_tensors="pt").input_ids
                    choice_ids = model.tokenizer(
                        choice, return_tensors="pt", add_special_tokens=False
                    ).input_ids
                    prompt_ids = prompt_ids.to(model.model.device)
                    choice_ids = choice_ids.to(model.model.device)
                    input_ids = torch.cat([prompt_ids, choice_ids], dim=-1)
                    labels = input_ids.clone()
                    labels[:, : prompt_ids.shape[1]] = -100
                    token_count = int((labels != -100).sum().item()) or 1
                    with torch.no_grad():
                        outputs = model.model(input_ids=input_ids, labels=labels)
                    loss = float(outputs.loss.item())
                    score = -loss if normalize == "mean" else -loss * token_count
                scores[choice] = score
            return scores

        scores = await asyncio.to_thread(lambda: [_score_prompt(p) for p in prompts])
        predictions: List[Any] = []
        extras: List[Dict[str, Any]] = []
        for prompt_scores in scores:
            best_choice = max(prompt_scores, key=prompt_scores.get)
            predicted_value: Any = best_choice
            if all(isinstance(item, int) for item in scale):
                predicted_value = int(best_choice)
            predictions.append(predicted_value)
            extras.append(
                {
                    "scale": scale,
                    "choice_logprobs": prompt_scores,
                    "predicted_choice": best_choice,
                }
            )
        return predictions, extras

    def normalize_for_metrics(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
        task_cfg: Mapping[str, Any],
    ) -> Tuple[List[Any], List[Any]]:
        """Return predictions/references unchanged for metric scoring."""
        return list(predictions), list(references)
