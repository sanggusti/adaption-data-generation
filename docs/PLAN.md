# Project Plan: adaption-data-generation

## Overview

A large-scale synthetic data generation pipeline for LLM finetuning, evaluation,
preference learning, and multimodal adaptation. Uses **pandas** for orchestration,
**vLLM / HF Hub** for inference, and the **Adaption SDK** for quality enhancement.
Output goes to **HuggingFace Datasets** and/or **Kaggle**.

```
Seeds ──► Pipelines ──► Quality ──► Exporters
          (SFT/DPO/    (Judge/      (HF Hub /
          Eval/Arena/  Dedup/       Kaggle)
          Multimodal)  Adaption)
               │
        Inference Backends
        (vLLM / HF Hub / Local)
```

---

## Phase 1 — Project Foundation & Infrastructure

### Issue 1: Project Scaffolding & Directory Structure
Set up a clean project layout:
```
src/
  adaption_datagen/
    schemas/       # Pydantic data models
    inference/     # vLLM and HF Hub backends
    pipelines/     # data generation pipelines
    enhancers/     # Adaption SDK integration
    exporters/     # HF Datasets, Kaggle exporters
    utils/         # shared utilities
configs/           # YAML configs per use-case
data/              # local staging area
scripts/           # CLI entry points / helper scripts
tests/
  unit/
  integration/
notebooks/
docs/
```
**Labels:** `infrastructure`, `good first issue`

---

### Issue 2: Dependency & Environment Setup
- `pyproject.toml` (hatchling build backend)
- Core deps: `vllm`, `transformers`, `datasets`, `pandas`, `pydantic`, `pydantic-settings`, `kaggle`, `openai`, `huggingface-hub`
- Optional extras: `[vllm]`, `[dev]`, `[docs]`
- Dev deps: `pytest`, `pytest-asyncio`, `ruff`, `mypy`
- `Dockerfile` + `docker-compose.yml` for reproducible inference environments
- `.env.example` with all required env vars (`HF_TOKEN`, `KAGGLE_KEY`, `VLLM_BASE_URL`, `ADAPTION_API_KEY`, etc.)

**Labels:** `infrastructure`

---

### Issue 3: Configuration Management
- YAML-based pipeline configs (model name, backend, prompt templates, output format, batch size)
- Pydantic config models with validation (`PipelineConfig`, `ModelConfig`, `GenerationConfig`, `EnhancementConfig`, `ExportConfig`)
- `pydantic-settings` for environment variable overrides
- `load_config(path)` utility function

**Labels:** `infrastructure`

---

## Phase 2 — Inference Backends

### Issue 4: vLLM / OpenAI-compatible Inference Backend
- `InferenceBackend` abstract base class with `generate()` and `generate_batch()` async methods
- `OpenAIBackend` implementation using the `openai` Python SDK
- Works transparently with vLLM (`--base_url`), OpenAI, and any OpenAI-compatible server
- Batched async generation with bounded concurrency (`asyncio.Semaphore`)
- Configurable sampling params: `temperature`, `top_p`, `max_tokens`, `stop`, `seed`

**Labels:** `inference`, `feature`

---

### Issue 5: HuggingFace Hub Inference Backend
- `HFBackend` with three modes:
  - `"api"` — HF Serverless Inference API via `huggingface_hub.AsyncInferenceClient`
  - `"endpoint"` — HF Dedicated Inference Endpoint (chat completion)
  - `"local"` — local `transformers` pipeline (runs in executor to avoid blocking event loop)
- Unified interface matching `InferenceBackend` base class

**Labels:** `inference`, `feature`

---

### Issue 6: Multi-Model Orchestrator
- `ModelOrchestrator` class: separate backend instances for `generator`, `judge`, `rewriter` roles
- Factory `_build_backend(cfg: ModelConfig)` to instantiate the right backend from config
- Retry logic with exponential backoff (`max_retries`, `retry_backoff_base`)
- Optional validation callback per generation

**Labels:** `inference`, `feature`

---

## Phase 3 — Core Data Generation Pipelines

### Issue 7: SFT Data Generation Pipeline
- Load seed topics/instructions from a file or inline YAML list
- Jinja2 prompt template rendering
- Multi-turn conversation generation (user/assistant format)
- Output formats: Alpaca, ShareGPT, OpenAI Chat
- Pydantic schemas: `AlpacaRecord`, `ShareGPTRecord`, `Message`
- Pandas DataFrame orchestration; JSONL output

**Labels:** `pipeline`, `feature`

---

### Issue 8: DPO / Preference Data Generation Pipeline
- Best-of-N sampling: generate N candidates per prompt at varied temperatures
- LLM-as-Judge scoring to select chosen/rejected pair
- Output schema: `DPORecord` (`prompt`, `chosen`, `rejected`)
- Configurable N and judge model

**Labels:** `pipeline`, `feature`

---

### Issue 9: LLM Evaluation Dataset Generation Pipeline
- Multiple-choice QA generation with distractor synthesis (`MCQARecord`)
- Open-ended benchmark generation: reasoning, coding, math (`EvalRecord`)
- Reference answer + rubric generation for judge-based eval
- `to_lm_eval_dict()` output compatible with `lm-evaluation-harness`

**Labels:** `pipeline`, `feature`

---

### Issue 10: Arena-Style Comparison Data Generation Pipeline
- Collect responses from N models for the same prompt
- LLM judge pairwise comparison with win/loss/tie
- ELO score tracking across samples
- `ArenaRecord` schema with `ModelResponse` list
- Output format compatible with Chatbot Arena / FastChat

**Labels:** `pipeline`, `feature`

---

### Issue 11: Multimodal Data Generation Pipeline
- Image captioning and VQA data generation using vision-language models (e.g., LLaVA, InternVL)
- Text-to-image prompt generation for diffusion model finetuning
- Image + text instruction pairs — LLaVA-style format (`MultimodalRecord`)
- Image source: local directory or HF dataset

**Labels:** `pipeline`, `feature`, `multimodal`

---

## Phase 4 — Data Quality & Enhancement

### Issue 12: Adaption SDK Integration
- `AdaptionEnhancer` class wrapping the Adaption SDK
- Quality scoring: call SDK to score each generated record
- Filtering: drop records below `min_quality_score` threshold
- Rewriting: optionally rewrite low-quality records using Adaption refinement API
- Configurable pipeline: score → filter → rewrite

**Labels:** `quality`, `feature`

---

### Issue 13: LLM-as-Judge Quality Scorer
- `LLMJudge` class: standalone scorer for any generated dataset
- Criteria: helpfulness, accuracy, safety, instruction-following
- Structured JSON output from judge, parsed into `QualityScore`
- Batch scoring with pandas DataFrame I/O
- Configurable rubric via Jinja2 template

**Labels:** `quality`, `feature`

---

### Issue 14: Deduplication & Filtering Pipeline
- MinHash near-duplicate detection (`datasketch`)
- Rule-based filters: min/max length, language detection (`langdetect`), regex blocklist
- Embedding-based semantic deduplication (`sentence-transformers` + cosine similarity)
- Pandas-native API: `deduplicate(df)`, `apply_filters(df, rules)`

**Labels:** `quality`, `feature`

---

### Issue 15: Prompt Template Manager
- Jinja2-based template rendering
- Template registry: built-in templates for SFT, DPO, eval, multimodal
- Load custom templates from file path in config
- Template versioning (name + version key)

**Labels:** `infrastructure`, `feature`

---

## Phase 5 — Output & Distribution

### Issue 16: HuggingFace Datasets Exporter
- `HFExporter`: push any generated dataset to HF Hub
- Auto-generate dataset card (README) from pipeline metadata
- Train/val/test splitting with configurable ratios
- Streaming upload for large datasets (`datasets.push_to_hub`)

**Labels:** `exporter`, `feature`

---

### Issue 17: Kaggle Datasets Exporter
- `KaggleExporter`: export to Parquet + `dataset-metadata.json`
- Create new dataset or update existing version via Kaggle API
- Auto-generate Kaggle metadata from pipeline config

**Labels:** `exporter`, `feature`

---

## Phase 6 — Scale & Performance

### Issue 18: Distributed Generation with Ray
- `DistributedPipeline`: wrap any pipeline with Ray for parallel execution
- Partition seed data across `num_workers` Ray actors
- Centralized result aggregation with pandas
- Progress tracking via Rich progress bar
- Checkpoint/resume: skip seeds already processed

**Labels:** `scale`, `feature`

---

### Issue 19: Caching & Checkpointing
- `CacheManager`: LLM response caching using `diskcache` (local) or Redis (distributed)
- Cache key: hash of (model, messages, sampling params)
- Pipeline-level checkpointing: save DataFrame state after each batch
- Incremental updates: detect new seeds and only generate for them

**Labels:** `scale`, `feature`

---

## Phase 7 — Tooling & Developer Experience

### Issue 20: CLI Interface
- `click`-based CLI entry point: `adaption-datagen`
- Sub-commands:
  - `generate --config <path>` — run a generation pipeline
  - `enhance --input <path> --output <path>` — run enhancement only
  - `export hf` / `export kaggle` — push an existing dataset
  - `evaluate --dataset <path>` — score a dataset with LLM judge
- Rich progress bars and summary tables

**Labels:** `dx`, `feature`

---

### Issue 21: Example Configs & Notebooks
- YAML configs for: `sft.yaml`, `dpo.yaml`, `eval.yaml`, `arena.yaml`, `multimodal.yaml`
- Jupyter notebooks:
  - `01_sft_quickstart.ipynb`
  - `02_dpo_preference.ipynb`
  - `03_eval_generation.ipynb`
  - `04_multimodal.ipynb`
  - `05_scale_with_ray.ipynb`
- Sample seed files in `data/seeds/`

**Labels:** `dx`, `documentation`

---

### Issue 22: CI/CD Pipeline
- GitHub Actions workflow `ci.yml`:
  - Lint with `ruff`
  - Type-check with `mypy`
  - Unit tests with `pytest` (no GPU required)
- Separate `publish.yml` workflow: build and publish to PyPI on tagged release

**Labels:** `infrastructure`, `ci`

---

### Issue 23: Comprehensive Documentation
- Update `README.md`: architecture diagram, quickstart, supported formats table
- `docs/` with MkDocs Material theme
- API reference (auto-generated with `mkdocstrings`)
- `CONTRIBUTING.md`: dev setup, coding style, PR process

**Labels:** `documentation`

---

## Milestones

| Milestone | Issues | Goal |
|-----------|--------|------|
| v0.1 — Foundation | 1, 2, 3, 22 | Installable package with CI |
| v0.2 — Inference | 4, 5, 6 | Both backends working |
| v0.3 — Pipelines | 7, 8, 9, 15 | SFT + DPO generation end-to-end |
| v0.4 — Quality | 12, 13, 14 | Enhancement + dedup working |
| v0.5 — Export | 16, 17, 20 | One-click HF Hub push + CLI |
| v0.6 — Full | 10, 11, 18, 19, 21, 23 | Arena, multimodal, Ray scale, docs |
