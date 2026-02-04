## space-eval-rag (RAG disabled for now)

This repo contains a minimal async pipeline to evaluate multiple LLMs over multiple tasks with pluggable metrics.

### Flow (current)

1. `main.py` loads a YAML config and instantiates `LLMEvaluator`.
2. `ConfigManager` validates config and fills defaults (batch size, concurrency, output dir).
3. `ModelFactory` creates model adapters (Hugging Face or vLLM; API stub exists).
4. For each model × task pair, `LLMEvaluator` schedules an async evaluation job.
5. Each job:
   - loads the dataset (`TaskProcessor.load_dataset`)
   - formats prompts with `prompt_template` + `input_mappings`
   - extracts references from `reference_field(s)`
   - runs `model.batch_generate(...)` in batches
   - computes metrics via `evaluate.load(...)` or a registered custom metric
   - writes a per-run JSON result to `evaluation.output_dir`
6. After all jobs finish, results are aggregated and a summary JSON is written.

### Quick start

1. Create a config based on `inference/config/sample-config.yaml`.
2. Run:

```bash
python main.py --config inference/config/sample-config.yaml
```

Outputs are written to `runs/`.

### Config highlights

- `models`: list of model configs (`type: huggingface | vllm | api`)
- `tasks`: list of tasks with dataset, prompt template, mappings, and metrics
- `evaluation`: batch size, max concurrency, output directory, sample size
