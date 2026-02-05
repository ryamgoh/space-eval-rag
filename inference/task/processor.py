from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import evaluate
from datasets import Dataset, load_dataset


class MetricRegistry:
    """Registry for custom metric functions."""
    _registry: Dict[str, Callable[..., Dict[str, float]]] = {}

    @classmethod
    def register(cls, name: str, fn: Callable[..., Dict[str, float]]) -> None:
        """Register a metric function by name."""
        cls._registry[name] = fn

    @classmethod
    def get(cls, name: str) -> Callable[..., Dict[str, float]] | None:
        """Fetch a registered metric function by name."""
        return cls._registry.get(name)


class TaskProcessor:
    """Utilities for dataset loading, templating, and metric computation."""
    @staticmethod
    def load_dataset(
        dataset_config: Any, split: str | None = None, cache_dir: Optional[str] = None
    ) -> Dataset:
        """Load a dataset from HF or a custom loader, with optional caching."""
        if isinstance(dataset_config, str):
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            return load_dataset(dataset_config, split=split or "validation", cache_dir=cache_dir)

        if not isinstance(dataset_config, dict):
            raise ValueError("dataset must be a string or dict.")

        if dataset_config.get("type") == "custom":
            loader_path = dataset_config["loader"]
            module_path, func_name = loader_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            loader = getattr(module, func_name)
            # Custom loaders may return a Dataset or any iterable of dicts.
            data = loader(dataset_config)
            if isinstance(data, Dataset):
                return data
            return Dataset.from_list(list(data))

        dataset_name = dataset_config.get("name") or dataset_config.get("path")
        if not dataset_name:
            raise ValueError("dataset config requires a name or path.")
        effective_cache_dir = dataset_config.get("cache_dir", cache_dir)
        if effective_cache_dir:
            os.makedirs(effective_cache_dir, exist_ok=True)
        return load_dataset(
            dataset_name,
            split=dataset_config.get("split", split or "validation"),
            cache_dir=effective_cache_dir,
            **dataset_config.get("kwargs", {}),
        )

    @staticmethod
    def apply_template(dataset: Dataset, template: str, mappings: Mapping[str, str]) -> List[str]:
        """Render prompt templates using dataset rows."""
        prompts: List[str] = []
        for row in dataset:
            values = {key: TaskProcessor.resolve_field(row, field) for key, field in mappings.items()}
            prompts.append(template.format_map(values))
        return prompts

    @staticmethod
    def extract_references(dataset: Dataset, task_config: Mapping[str, Any]) -> List[Any]:
        """Extract reference labels/targets from dataset rows."""
        ref_field = task_config.get("reference_field")
        if isinstance(ref_field, str):
            return [TaskProcessor.resolve_field(row, ref_field) for row in dataset]
        ref_fields = task_config.get("reference_fields")
        if isinstance(ref_fields, Sequence) and not isinstance(ref_fields, (str, bytes)):
            return [
                {field: TaskProcessor.resolve_field(row, field) for field in ref_fields}
                for row in dataset
            ]
        raise ValueError("Task must specify reference_field or reference_fields.")

    @staticmethod
    def resolve_field(row: Mapping[str, Any], path: str) -> Any:
        """Resolve dot-path fields (with optional list indices) from a row."""
        current: Any = row
        for part in path.split("."):
            if isinstance(current, Mapping):
                if part not in current:
                    raise KeyError(f"Missing field '{part}' in path '{path}'.")
                current = current[part]
                continue
            if isinstance(current, list):
                try:
                    index = int(part)
                except ValueError as exc:
                    raise KeyError(f"Expected list index in path '{path}', got '{part}'.") from exc
                if index >= len(current) or index < 0:
                    raise IndexError(f"Index {index} out of range for path '{path}'.")
                current = current[index]
                continue
            raise KeyError(f"Cannot resolve '{part}' in path '{path}'.")
        return current

    @staticmethod
    def postprocess_predictions(predictions: List[str], task_config: Mapping[str, Any]) -> List[str]:
        """Apply simple string post-processing rules to predictions."""
        postprocess = task_config.get("prediction_postprocess") or {}
        thinking_cfg = task_config.get("thinking_delimiters") or {}
        strip_thinking = bool(thinking_cfg.get("strip_from_prediction"))

        processed: List[str] = []
        for pred in predictions:
            text = str(pred)
            if strip_thinking:
                # Remove thinking segment before other postprocessing steps.
                _, text = TaskProcessor.apply_thinking_delimiters(text, thinking_cfg)
            after = postprocess.get("after")
            before = postprocess.get("before")
            if after and after in text:
                text = text.split(after)[-1]
            if before and before in text:
                text = text.split(before)[0]
            processed.append(text.strip())
        return processed

    @staticmethod
    def apply_thinking_delimiters(
        text: str, delimiters: Mapping[str, Any]
    ) -> tuple[Optional[str], str]:
        """Extract thinking content and return the answer based on delimiter config."""
        start = delimiters.get("start")
        end = delimiters.get("end")
        if not start and not end:
            return None, text
        if start and start not in text:
            return None, text
        if end and end not in text:
            return None, text

        if start:
            before_start, after_start = text.split(start, 1)
            if end and end in after_start:
                thinking_body, after_end = after_start.split(end, 1)
            else:
                thinking_body, after_end = after_start, ""
            keep = bool(delimiters.get("keep_delimiters"))
            thinking = f"{start}{thinking_body}{end}" if keep and end else thinking_body

            mode = delimiters.get("mode", "after_end")
            if mode == "after_end":
                answer = after_end
            elif mode == "remove":
                answer = f"{before_start}{after_end}"
            elif mode == "before_start":
                answer = before_start
            else:
                answer = text
            return thinking.strip(), answer.strip()

        # End-only delimiters: split on the last end token.
        before_end, after_end = text.rsplit(end, 1)
        keep = bool(delimiters.get("keep_delimiters"))
        thinking = f"{before_end}{end}" if keep else before_end
        mode = delimiters.get("mode", "after_end")
        if mode == "after_end":
            answer = after_end
        elif mode == "remove":
            answer = after_end
        elif mode == "before_start":
            answer = before_end
        else:
            answer = text
        return thinking.strip(), answer.strip()

    @staticmethod
    def normalize_classification(
        predictions: List[str],
        references: List[Any],
        label_map: Mapping[Any, str] | Sequence[str],
    ) -> tuple[List[int], List[int]]:
        """Normalize classification predictions/refs to numeric label IDs."""
        label_list: List[str]
        label_to_id: Dict[str, int] = {}
        id_to_label: Dict[int, str] = {}

        if isinstance(label_map, Mapping):
            for key, value in label_map.items():
                if isinstance(key, (int, float)) and isinstance(value, str):
                    idx = int(key)
                    id_to_label[idx] = value
                    label_to_id[value.lower()] = idx
                elif isinstance(key, str) and isinstance(value, (int, float)):
                    idx = int(value)
                    id_to_label[idx] = key
                    label_to_id[key.lower()] = idx
            if not id_to_label:
                raise ValueError("label_map must map ids to labels or labels to ids.")
            label_list = [label for _, label in sorted(id_to_label.items())]
        else:
            label_list = list(label_map)
            id_to_label = {idx: label for idx, label in enumerate(label_list)}
            label_to_id = {label.lower(): idx for idx, label in id_to_label.items()}

        normalized_refs: List[int] = []
        for ref in references:
            if isinstance(ref, (int, float)) and int(ref) in id_to_label:
                normalized_refs.append(int(ref))
            else:
                ref_text = str(ref).strip().lower()
                normalized_refs.append(label_to_id.get(ref_text, -1))

        normalized_preds: List[int] = []
        for pred in predictions:
            pred_text = str(pred).strip()
            pred_lower = pred_text.lower()
            matched_id = None
            if pred_lower in label_to_id:
                matched_id = label_to_id[pred_lower]
            else:
                for label in label_list:
                    if label.lower() in pred_lower:
                        matched_id = label_to_id[label.lower()]
                        break
            if matched_id is None:
                try:
                    matched_id = int(pred_text)
                except ValueError:
                    matched_id = -1
            normalized_preds.append(matched_id)

        return normalized_preds, normalized_refs

    @staticmethod
    def compute_metrics(
        predictions: List[str],
        references: Iterable[Any],
        metrics: Iterable[Any],
    ) -> Dict[str, float]:
        """Compute metrics using registered custom or HF evaluate metrics."""
        results: Dict[str, float] = {}
        for metric_spec in metrics:
            if isinstance(metric_spec, str):
                name = metric_spec
                metric_args: Dict[str, Any] = {}
            else:
                name = metric_spec["name"]
                metric_args = metric_spec.get("args", {})

            custom = MetricRegistry.get(name)
            if custom:
                metric_result = custom(predictions=predictions, references=references, **metric_args)
            else:
                try:
                    metric = evaluate.load(name)
                    metric_result = metric.compute(predictions=predictions, references=references, **metric_args)
                except Exception as exc:  # pragma: no cover - fallback
                    raise ValueError(f"Failed to compute metric '{name}': {exc}") from exc

            for key, value in metric_result.items():
                results[f"{name}:{key}"] = value
        return results
