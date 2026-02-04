from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from inference.config.manager import ConfigManager
from inference.model.base_model import BaseModel
from inference.model.model import ModelFactory
from inference.task.aggregator import ResultAggregator
from inference.task.processor import TaskProcessor
from inference.util.parallel_executor import ParallelExecutor


class LLMEvaluator:
    def __init__(self, config_path: str):
        self.config = ConfigManager.load_yaml(config_path)
        self.models = self._load_models()
        self.tasks = self._load_tasks()

    def _load_models(self) -> List[BaseModel]:
        return [ModelFactory.get_model(model_cfg) for model_cfg in self.config["models"]]

    def _load_tasks(self) -> List[Dict[str, Any]]:
        return list(self.config["tasks"])

    def run(self) -> Dict[str, Any]:
        return asyncio.run(self.run_async())

    async def run_async(self) -> Dict[str, Any]:
        evaluation = self.config["evaluation"]
        executor = ParallelExecutor(evaluation["max_concurrent"])
        coroutines = [
            self._evaluate_single(model, task)
            for model in self.models
            for task in self.tasks
        ]
        results = await executor.run(coroutines)
        aggregated = ResultAggregator.aggregate_results(results)
        output_dir = Path(evaluation["output_dir"])
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ResultAggregator.save_results(
            aggregated,
            str(output_dir / f"summary_{timestamp}.json"),
            format="json",
        )
        return aggregated

    async def _evaluate_single(self, model: BaseModel, task: Dict[str, Any]) -> Dict[str, Any]:
        evaluation = self.config["evaluation"]
        dataset = TaskProcessor.load_dataset(task["dataset"], split=task.get("split"))
        sample_size = task.get("sample_size", evaluation.get("sample_size"))
        if sample_size:
            dataset = dataset.select(range(min(sample_size, len(dataset))))

        prompts = TaskProcessor.apply_template(dataset, task["prompt_template"], task["input_mappings"])
        references = TaskProcessor.extract_references(dataset, task)

        generation_kwargs = task.get("generation_kwargs", {})
        predictions = await model.batch_generate(
            prompts,
            batch_size=evaluation["batch_size"],
            **generation_kwargs,
        )

        predictions = TaskProcessor.postprocess_predictions(predictions, task)
        label_map = task.get("label_map")
        if label_map:
            predictions, references = TaskProcessor.normalize_classification(
                predictions, references, label_map
            )

        metrics = TaskProcessor.compute_metrics(predictions, references, task["metrics"])
        prediction_sample_size = evaluation.get("prediction_sample_size", 5)

        result = {
            "model": model.name,
            "task": task["name"],
            "metrics": metrics,
            "num_examples": len(prompts),
            "prediction_samples": predictions[:prediction_sample_size],
        }

        output_dir = Path(evaluation["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        ResultAggregator.save_results(
            {"runs": [result]},
            str(output_dir / f"{task['name']}__{model.name}.json"),
            format="json",
        )
        return result
