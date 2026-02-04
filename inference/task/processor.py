from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import evaluate
from datasets import Dataset, load_dataset


class MetricRegistry:
    _registry: Dict[str, Callable[..., Dict[str, float]]] = {}

    @classmethod
    def register(cls, name: str, fn: Callable[..., Dict[str, float]]) -> None:
        cls._registry[name] = fn

    @classmethod
    def get(cls, name: str) -> Callable[..., Dict[str, float]] | None:
        return cls._registry.get(name)


class TaskProcessor:
    @staticmethod
    def load_dataset(dataset_config: Any, split: str | None = None) -> Dataset:
        if isinstance(dataset_config, str):
            return load_dataset(dataset_config, split=split or "validation")

        if not isinstance(dataset_config, dict):
            raise ValueError("dataset must be a string or dict.")

        if dataset_config.get("type") == "custom":
            loader_path = dataset_config["loader"]
            module_path, func_name = loader_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            loader = getattr(module, func_name)
            data = loader(dataset_config)
            if isinstance(data, Dataset):
                return data
            return Dataset.from_list(list(data))

        dataset_name = dataset_config.get("name") or dataset_config.get("path")
        if not dataset_name:
            raise ValueError("dataset config requires a name or path.")
        return load_dataset(dataset_name, split=dataset_config.get("split", split or "validation"), **dataset_config.get("kwargs", {}))

    @staticmethod
    def apply_template(dataset: Dataset, template: str, mappings: Mapping[str, str]) -> List[str]:
        prompts: List[str] = []
        for row in dataset:
            values = {key: row[field] for key, field in mappings.items()}
            prompts.append(template.format_map(values))
        return prompts

    @staticmethod
    def extract_references(dataset: Dataset, task_config: Mapping[str, Any]) -> List[Any]:
        ref_field = task_config.get("reference_field")
        if isinstance(ref_field, str):
            return [row[ref_field] for row in dataset]
        ref_fields = task_config.get("reference_fields")
        if isinstance(ref_fields, Sequence) and not isinstance(ref_fields, (str, bytes)):
            return [{field: row[field] for field in ref_fields} for row in dataset]
        raise ValueError("Task must specify reference_field or reference_fields.")

    @staticmethod
    def postprocess_predictions(predictions: List[str], task_config: Mapping[str, Any]) -> List[str]:
        postprocess = task_config.get("prediction_postprocess")
        if not postprocess:
            return predictions

        after = postprocess.get("after")
        before = postprocess.get("before")
        processed: List[str] = []
        for pred in predictions:
            text = str(pred)
            if after and after in text:
                text = text.split(after)[-1]
            if before and before in text:
                text = text.split(before)[0]
            processed.append(text.strip())
        return processed

    @staticmethod
    def normalize_classification(
        predictions: List[str],
        references: List[Any],
        label_map: Mapping[Any, str] | Sequence[str],
    ) -> tuple[List[int], List[int]]:
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
