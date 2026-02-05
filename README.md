## space-eval-rag (RAG disabled for now)

Minimal async pipeline to evaluate multiple LLMs across multiple tasks with pluggable metrics.

### Flow (current)

1. `main.py` loads a YAML config and instantiates `LLMEvaluator`.
2. `ConfigManager` validates config and fills defaults (batch size, concurrency, output dir, cache dirs).
3. `ModelFactory` creates model adapters (Hugging Face or vLLM; API stub exists).
4. For each model × task pair, `LLMEvaluator` schedules an async evaluation job.
5. Each job:
   - loads the dataset (`TaskProcessor.load_dataset`)
   - formats prompts with `prompt_template(s)` + `input_mappings` (or `fields`)
   - extracts references from `reference_field(s)` when provided
   - runs `model.batch_generate(...)` in batches
   - computes metrics via `evaluate.load(...)` or a registered custom metric
   - writes a per-task JSON result under a run folder
6. After all jobs finish, results are aggregated and a run summary is written.

### Quick start

1. Create a config based on `inference/config/sample_config.yaml`.
2. Run:

```bash
python main.py --config inference/config/sample_config.yaml
```

Outputs are written to `runs/<run_id>/`.

### Output layout

```
runs/
  <run_id>/
    summary.json
    <model_name>/
      <task_name>.json
```

Detailed examples (when enabled) include:
`prediction_raw`, `prediction_parsed`, `prediction`, `actual_raw`, `actual`.

### Config highlights

- `models`: list of model configs (`type: huggingface | vllm | api`)
- `tasks`: list of tasks with dataset, prompt template, mappings, and metrics
- `evaluation`: batch size, max concurrency, output directory, sample size, and output detail controls
- `datasets_dir`: where HF datasets are cached (default: `./datasets`)
- `models_dir`: base directory for model downloads (default: `./models`)

### Generic task adapter

The default `generic` adapter supports:
- prompt templating from dataset fields
- reference extraction
- optional post-processing (`prediction_postprocess`)
- optional classification normalization (`label_map`)

Dot-paths are supported in mappings and references (e.g., `translation.en`, `answers.0.text`).

### Model caching

If `local_dir` is omitted on a model, it is automatically set to:

```
models_dir/<model_path>
```

Example:

```
models_dir: "./models"
model_path: "google/flan-t5-small"
```

becomes:

```
./models/google/flan-t5-small
```

### Example configs in this repo

- `inference/config/sample_config.yaml` (minimal default)
- `inference/config/tiny_test_config.yaml` (small multi-model sanity test)
- `inference/config/qa_classify.yaml` (QA + classification)
- `inference/config/classify.yaml` (classification only)
- `inference/config/sciq_generic.yaml` (SciQ QA test)
- `inference/config/generic_test.yaml` (generic adapter demo)

### Notes

- Metrics are optional; set `metrics: []` to skip scoring.
- API models are stubbed; vLLM requires its dependency installed.
