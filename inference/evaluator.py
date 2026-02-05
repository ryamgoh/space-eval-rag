from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from inference.adapters import metric_adapters, task_adapters
from inference.config.manager import ConfigManager
from inference.model.base_model import BaseModel
from inference.model.model import ModelFactory
from inference.task.aggregator import ResultAggregator
from inference.task.processor import TaskProcessor
from inference.util.parallel_executor import ParallelExecutor


class LLMEvaluator:
    """Run multi-model, multi-task evaluations using async execution."""
    def __init__(self, config_path: str):
        """Load config and initialize model/task lists."""
        self.config = ConfigManager.load_yaml(config_path)
        self.models = self._load_models()
        self.tasks = self._load_tasks()

    def _load_models(self) -> List[BaseModel]:
        """Instantiate model adapters from config."""
        return [ModelFactory.get_model(model_cfg) for model_cfg in self.config["models"]]

    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Return task configs as a list."""
        return list(self.config["tasks"])

    def run(self) -> Dict[str, Any]:
        """Run the evaluation synchronously by wrapping the async runner."""
        return asyncio.run(self.run_async())

    async def run_async(self) -> Dict[str, Any]:
        """Execute evaluation jobs concurrently and write summary outputs."""
        evaluation = self.config["evaluation"]
        executor = ParallelExecutor(evaluation["max_concurrent"])
        output_dir = Path(evaluation["output_dir"])
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        coroutines = [
            self._evaluate_single(model, task, run_dir)
            for model in self.models
            for task in self.tasks
        ]
        results = await executor.run(coroutines)
        aggregated = ResultAggregator.aggregate_results(results)
        aggregated["run_id"] = run_id
        ResultAggregator.save_results(
            aggregated,
            str(run_dir / "summary.json"),
            format="json",
        )
        # ResultAggregator.save_results(
        #     aggregated,
        #     str(output_dir / f"summary_{run_id}.json"),
        #     format="json",
        # )
        return aggregated

    async def _evaluate_single(
        self, model: BaseModel, task: Dict[str, Any], run_dir: Path
    ) -> Dict[str, Any]:
        """Evaluate a single model-task pair and return structured results."""
        evaluation = self.config["evaluation"]
        dataset = TaskProcessor.load_dataset(
            task["dataset"],
            split=task.get("split"),
            cache_dir=self.config.get("datasets_dir"),
        )
        sample_size = task.get("sample_size", evaluation.get("sample_size"))
        if sample_size:
            dataset = dataset.select(range(min(sample_size, len(dataset))))

        # Select adapter based on explicit override or task shape.
        if task.get("task_adapter"):
            adapter_name = task["task_adapter"]
        else:
            adapter_name = "generic"
        task_adapter = task_adapters.create(adapter_name)

        prompts = task_adapter.build_prompts(dataset, task)
        references = task_adapter.extract_references(dataset, task)

        generation_kwargs = task.get("generation_kwargs", {})
        completion_predictions, raw_predictions, extras = await task_adapter.generate_predictions(
            model,
            prompts,
            task,
            batch_size=evaluation["batch_size"],
            **generation_kwargs,
        )

        postprocessed_predictions = task_adapter.postprocess_predictions(completion_predictions, task)
        metric_predictions, metric_references = task_adapter.normalize_for_metrics(
            postprocessed_predictions, references, task
        )

        metrics = {}
        for metric_cfg in task.get("metrics", []):
            adapter = "hf"
            if isinstance(metric_cfg, dict):
                adapter = metric_cfg.get("adapter", "hf")
            metric_adapter = metric_adapters.create(adapter)
            metrics.update(
                metric_adapter.compute(metric_predictions, metric_references, metric_cfg)
            )
        prediction_sample_size = evaluation.get("prediction_sample_size", 5)

        result = {
            "model": model.name,
            "task": task["name"],
            "task_adapter": adapter_name,
            "metrics": metrics,
            "num_examples": len(prompts),
            "prediction_samples": metric_predictions[:prediction_sample_size],
        }

        if evaluation.get("save_detailed"):
            max_examples = evaluation.get("max_detailed_examples")
            total = len(prompts)
            limit = total if max_examples is None else min(total, max_examples)
            if extras is None:
                extras = task_adapter.collect_extras(task, limit) or []
            detailed_examples = []
            for idx in range(limit):
                extra = extras[idx] if extras else {}
                prediction_with_prompt = raw_predictions[idx]
                detailed_examples.append(
                    {
                        "index": idx,
                        "prompt": prompts[idx],
                        "prediction_raw": raw_predictions[idx],
                        "prediction_parsed": postprocessed_predictions[idx],
                        "prediction": metric_predictions[idx],
                        "prediction_with_prompt": prediction_with_prompt,
                        "actual_raw": references[idx],
                        "actual": metric_references[idx],
                        "extra": extra,
                    }
                )
            result["examples"] = detailed_examples

        model_dir_name = model.name.replace("/", "_").replace(" ", "_")
        model_dir = run_dir / model_dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        ResultAggregator.save_results(
            {"runs": [result]},
            str(model_dir / f"{task['name']}.json"),
            format="json",
        )
        return result
