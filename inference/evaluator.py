from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from inference.adapters import metric_adapters, task_adapters
from inference.analysis import AnalysisManager
from inference.config.manager import ConfigManager
from inference.logging.logger import Monitor
from inference.model.base_model import BaseModel
from inference.model.factory import ModelFactory
from inference.task.aggregator import ResultAggregator
from inference.task.processor import TaskProcessor
from inference.util.parallel_executor import ParallelExecutor


class LLMEvaluator:
    """Run multi-model, multi-task evaluations using async execution."""
    def __init__(self, config_path: str):
        """Load config and initialize model/task lists."""
        self.config = ConfigManager.load_yaml(config_path)
        self.analysis = AnalysisManager(self.config.get("analysis"))
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
        # Centralized progress/metrics logging.
        self.monitor = Monitor(
            enabled=bool(evaluation.get("log_progress")),
            level=str(evaluation.get("log_level", "INFO")),
        )
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
        await asyncio.to_thread(self.analysis.finalize_run, run_dir)
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
        metrics_cfgs = task.get("metrics", [])
        has_metrics = bool(metrics_cfgs)
        has_reference_spec = "reference_field" in task or "reference_fields" in task
        if has_metrics or (evaluation.get("save_detailed") and has_reference_spec):
            references = task_adapter.extract_references(dataset, task)
        else:
            references = []

        model_dir_name = model.name.replace("/", "_").replace(" ", "_")
        model_dir = run_dir / model_dir_name
        checkpoint_batches = evaluation.get("checkpoint_batches") or 0
        pending_rows: List[Dict[str, Any]] = []
        if checkpoint_batches and checkpoint_batches > 0:
            model_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = model_dir / f"{task['name']}.jsonl"
            checkpoint_path.write_text("", encoding="utf-8")

            def _flush_checkpoint() -> None:
                if not pending_rows:
                    return
                with checkpoint_path.open("a", encoding="utf-8") as handle:
                    for row in pending_rows:
                        handle.write(json.dumps(row) + "\n")
                pending_rows.clear()

            def _batch_cb(
                batch_index: int,
                total_batches: int,
                batch_start: int,
                batch_prompts: List[str],
                batch_outputs: List[str],
                batch_raw: List[str] | None,
            ) -> None:
                raw_list = batch_raw if batch_raw is not None else list(batch_outputs)
                parsed = task_adapter.postprocess_predictions(batch_outputs, task)
                for offset, (prompt, raw, parsed_text) in enumerate(
                    zip(batch_prompts, raw_list, parsed)
                ):
                    idx = batch_start + offset
                    actual_raw = references[idx] if idx < len(references) else None
                    pending_rows.append(
                        {
                            "model": model.name,
                            "task": task["name"],
                            "index": idx,
                            "prompt": prompt,
                            "prediction_raw": raw,
                            "prediction_parsed": parsed_text,
                            "actual_raw": actual_raw,
                        }
                    )
                if batch_index == total_batches or batch_index % checkpoint_batches == 0:
                    _flush_checkpoint()
        else:
            _batch_cb = None

        generation_kwargs = task.get("generation_kwargs", {})
        log_progress = bool(evaluation.get("log_progress"))
        progress_every = evaluation.get("progress_every_batches", 1)

        def _progress_cb(done: int, total: int) -> None:
            # Emit per-batch progress via the shared monitor.
            if self.monitor:
                self.monitor.log_progress(model.name, task["name"], done, total)

        completion_predictions, raw_predictions, extras = await task_adapter.generate_predictions(
            model,
            prompts,
            task,
            batch_size=evaluation["batch_size"],
            batch_cb=_batch_cb,
            progress_cb=_progress_cb if log_progress else None,
            progress_every=progress_every,
            **generation_kwargs,
        )

        postprocessed_predictions = task_adapter.postprocess_predictions(completion_predictions, task)
        if has_metrics:
            metric_predictions, metric_references = task_adapter.normalize_for_metrics(
                postprocessed_predictions, references, task
            )
        else:
            metric_predictions = list(postprocessed_predictions)
            metric_references = list(references)

        metrics = {}
        for metric_cfg in metrics_cfgs:
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
        model_special_tokens = model.get_special_tokens()
        if model_special_tokens:
            # Preserve model-specific special token metadata in the run output.
            result["model_special_tokens"] = model_special_tokens

        if evaluation.get("save_detailed"):
            max_examples = evaluation.get("max_detailed_examples")
            total = len(prompts)
            limit = total if max_examples is None else min(total, max_examples)
            if extras is None:
                extras = task_adapter.collect_extras(task, limit) or []
            thinking_cfg = task.get("thinking_delimiters") or {}
            detailed_examples = []
            for idx in range(limit):
                extra = extras[idx] if extras else {}
                prediction_with_prompt = raw_predictions[idx]
                thinking = None
                answer_from_thinking = None
                if thinking_cfg:
                    # Split thinking/answer segments for inspection.
                    thinking, answer_from_thinking = TaskProcessor.apply_thinking_delimiters(
                        completion_predictions[idx], thinking_cfg
                    )
                actual_raw = references[idx] if references else None
                actual = metric_references[idx] if metric_references else None
                detailed_examples.append(
                    {
                        "index": idx,
                        "prompt": prompts[idx],
                        "prediction_raw": raw_predictions[idx],
                        "prediction_parsed": postprocessed_predictions[idx],
                        "prediction": metric_predictions[idx],
                        "prediction_with_prompt": prediction_with_prompt,
                        "prediction_thinking": thinking,
                        "prediction_answer": answer_from_thinking,
                        "actual_raw": actual_raw,
                        "actual": actual,
                        "extra": extra,
                    }
                )
            result["examples"] = detailed_examples

        model_dir.mkdir(parents=True, exist_ok=True)
        ResultAggregator.save_results(
            {"runs": [result]},
            str(model_dir / f"{task['name']}.json"),
            format="json",
        )
        await asyncio.to_thread(self.analysis.update_incremental, run_dir)
        return result
