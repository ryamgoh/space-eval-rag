## space-eval-rag

Minimal async pipeline to evaluate multiple LLMs across multiple tasks with pluggable metrics.

### Flow (current)

1. `main.py` loads a YAML config and instantiates `LLMEvaluator`.
2. `ConfigManager` validates config and fills defaults (batch size, concurrency, output dir).
3. `ModelFactory` creates model adapters (Hugging Face or vLLM; API stub exists). Adapters live in `inference/model/`.
4. For each model × task pair, `LLMEvaluator` schedules an async evaluation job.
5. Each job:
   - loads the dataset (`TaskProcessor.load_dataset`)
   - formats prompts with `prompt_template` + `input_mappings`
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
`prediction_raw`, `prediction_parsed`, `prediction`, `prediction_with_prompt`,
`prediction_thinking`, `prediction_answer`, `actual_raw`, `actual`.

Notes:
- `prediction_raw` preserves special tokens (e.g., `<pad>`, `</s>`).
- `prediction` and `prediction_parsed` are cleaned of special tokens; for causal models they are the completion only.
- `prediction_with_prompt` mirrors `prediction_raw` so you can inspect the full model output.
- `model_special_tokens` is included in run outputs when the backend exposes tokenizer metadata (HF models).

### Config highlights

- `models`: list of model configs (`type: huggingface | vllm | api`)
- `tasks`: list of tasks with dataset, `prompt_template`, `input_mappings`, and metrics
- `evaluation`: batch size, max concurrency, output directory, sample size, output detail controls, and optional per-batch logging (`log_progress`, `progress_every_batches`)
- `datasets_dir`: where HF datasets are cached (default: `./datasets`)
- `models_dir`: base directory for model downloads (default: `./models`)
- `rag`: optional per-task block to enable retrieval-augmented prompts

RAG can be enabled per task, so a single config can mix tasks that use retrieval with tasks that do not.

### Text generation adapter

The default `text_generation` adapter (also available as `generic`) supports:
- prompt templating from dataset fields via `input_mappings` and `prompt_template`
- reference extraction
- optional post-processing (`prediction_postprocess`)
- optional classification normalization (`label_map`)
- optional thinking delimiter extraction (`thinking_delimiters`)
- optional RAG context injection (`rag`)

Additional adapters are available for clarity when configuring tasks:
- `classification`
- `qa`
- `summarization`

Dot-paths are supported in mappings and references (e.g., `translation.en`, `answers.0.text`).

### RAG (FAISS + embeddings)

You can enable a simple retrieval-augmented flow on a per-task basis. RAG builds a
FAISS index from a separate corpus dataset, retrieves top-k passages, and injects
them into the prompt via a `{context}` placeholder.

Example:

```yaml
tasks:
  - name: "qa-sciq-rag"
    dataset: "sciq"
    split: "validation"
    prompt_template: |
      Use the context to answer.
      Context:
      {context}

      Question: {question}
      Answer:
    input_mappings:
      question: "question"
    reference_field: "correct_answer"
    rag:
      enabled: true
      context_k: 3
      context_template: "- {text}"
      context_separator: "\n"
      query_template: "Question: {question}"
      query_mappings:
        question: "question"
      corpus:
        name: "sciq"
        split: "train"
      corpus_template: "{question}\n{correct_answer}"
      corpus_mappings:
        question: "question"
        correct_answer: "correct_answer"
      embedding_model:
        model_path: "sentence-transformers/all-MiniLM-L6-v2"
        local_dir: "./models/all-MiniLM-L6-v2"
        max_length: 256
        batch_size: 32
        pooling: "mean"
```

Notes:
- `{context}` must appear in the `prompt_template` when RAG is enabled.
- `rag.query_template` defaults to the task's `prompt_template` with `{context}` removed.

#### File corpus loader (LangChain)

You can point `rag.corpus` at local files directly:

```yaml
rag:
  enabled: true
  corpus:
    type: "files"
    paths:
      - "/path/to/docs"
      - "/path/to/notes.md"
    glob: "data/**/*.jsonl"
    recursive: true
    extensions: ["txt", "md", "jsonl", "pdf"]
    chunk_size: 1200
    chunk_overlap: 200
    json_content_key: "text"
    # json_jq_schema: ".text"
  corpus_template: "{text}"
  corpus_mappings:
    text: "text"
  embedding_model:
    model_path: "sentence-transformers/all-MiniLM-L6-v2"
```

Supported formats: `.txt`, `.md`, `.jsonl`, `.pdf`. JSONL uses LangChain's
`JSONLoader` (install the `jq` Python package). JSONL defaults to the `text`
field; override with `json_content_key` or `json_jq_schema`.
Chunking uses LangChain's `RecursiveCharacterTextSplitter` when `chunk_size` is provided.

### Thinking delimiters

You can split or strip "thinking" content with task-level config:

```yaml
thinking_delimiters:
  start: "<think>"
  end: "</think>"
  mode: "after_end"          # after_end | remove | before_start | none
  strip_from_prediction: true
```

When enabled, detailed examples include `prediction_thinking` and `prediction_answer`.

End-only example (models that emit only a closing tag):

```yaml
thinking_delimiters:
  end: "</thinking>"
  mode: "after_end"
  strip_from_prediction: true
```

For end-only delimiters, the split uses the last occurrence of `end`.

### Device placement (HF vs vLLM)

Hugging Face models are explicitly moved to a device. The adapter uses `device: "cuda"` when available (falls back to CPU). You can override it per model:

```yaml
models:
  - name: "flan-t5-small"
    type: "huggingface"
    model_path: "google/flan-t5-small"
    device: "cuda"
```

If you provide `model_kwargs.device_map`, we skip manual `.to(...)` so HF/Accelerate can shard/offload the model:

```yaml
models:
  - name: "big-model"
    type: "huggingface"
    model_path: "org/model"
    model_kwargs:
      device_map: "auto"
```

vLLM manages device placement internally and uses visible GPUs. You should control GPU visibility via your environment (e.g., `CUDA_VISIBLE_DEVICES`) or vLLM’s own config when exposed.

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
- `inference/config/rag_example.yaml` (RAG + FAISS demo)

### Notes

- Metrics are optional; set `metrics: []` to skip scoring.
- When `metrics: []`, reference fields are optional unless `save_detailed` is enabled.
- API models are stubbed; vLLM requires its dependency installed.
