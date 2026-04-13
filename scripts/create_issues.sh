#!/usr/bin/env bash
# create_issues.sh
# Creates all 23 project issues in the adaption-data-generation repo.
# Prerequisites: `gh` CLI installed and authenticated (`gh auth login`)
# Usage: bash scripts/create_issues.sh

set -euo pipefail

REPO="sanggusti/adaption-data-generation"

echo "Creating issues in $REPO ..."

# ─── Phase 1: Foundation & Infrastructure ────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "Project Scaffolding & Directory Structure" \
  --label "infrastructure,good first issue" \
  --body "## Summary
Set up the clean project layout for the package.

## Directory structure
\`\`\`
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
scripts/           # helper scripts
tests/unit/
tests/integration/
notebooks/
docs/
\`\`\`

## Tasks
- [ ] Create all directories with \`__init__.py\` stubs
- [ ] Add \`src/adaption_datagen/_version.py\`
- [ ] Add top-level \`LICENSE\` (MIT)
- [ ] Add \`.gitignore\` additions (data/, .cache/, .checkpoints/)

Ref: [docs/PLAN.md](docs/PLAN.md)"

gh issue create --repo "$REPO" \
  --title "Dependency & Environment Setup" \
  --label "infrastructure" \
  --body "## Summary
Set up package metadata, dependency management, and containerisation.

## Tasks
- [ ] \`pyproject.toml\` with hatchling build backend
- [ ] Core deps: \`vllm\`, \`transformers\`, \`datasets\`, \`pandas\`, \`pydantic\`, \`pydantic-settings\`, \`kaggle\`, \`openai\`, \`huggingface-hub\`, \`ray[default]\`, \`diskcache\`, \`datasketch\`, \`jinja2\`, \`click\`, \`rich\`
- [ ] Optional extras: \`[vllm]\`, \`[dev]\`, \`[docs]\`
- [ ] Dev deps: \`pytest\`, \`pytest-asyncio\`, \`ruff\`, \`mypy\`
- [ ] \`Dockerfile\` (CUDA base image for vLLM)
- [ ] \`docker-compose.yml\` (vLLM server + pipeline runner services)
- [ ] \`.env.example\` with all required env vars

Ref: [docs/PLAN.md](docs/PLAN.md)"

gh issue create --repo "$REPO" \
  --title "Configuration Management (Pydantic + YAML)" \
  --label "infrastructure" \
  --body "## Summary
Centralised, validated configuration with YAML input and env-var overrides.

## Tasks
- [ ] \`PipelineConfig\`, \`ModelConfig\`, \`GenerationConfig\`, \`EnhancementConfig\`, \`OutputConfig\`, \`ExportConfig\`, \`CacheConfig\`, \`DistributedConfig\` Pydantic models
- [ ] \`MultiModelConfig\`: generator / judge / rewriter roles
- [ ] \`Settings\` class via \`pydantic-settings\` for env overrides
- [ ] \`load_config(path)\` utility
- [ ] Unit tests for validation and env override

Ref: [docs/PLAN.md](docs/PLAN.md)"

# ─── Phase 2: Inference Backends ─────────────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "vLLM / OpenAI-compatible Inference Backend" \
  --label "inference,feature" \
  --body "## Summary
Async inference backend that works with vLLM, OpenAI, and any OpenAI-compatible server.

## Tasks
- [ ] \`InferenceBackend\` abstract base class (\`generate\`, \`generate_batch\`)
- [ ] \`GenerationRequest\` and \`GenerationResult\` dataclasses
- [ ] \`OpenAIBackend\` using \`openai.AsyncOpenAI\`
- [ ] \`generate_batch\` with bounded concurrency (\`asyncio.Semaphore\`)
- [ ] Configurable: temperature, top_p, max_tokens, stop, seed
- [ ] Unit tests with mocked responses

Ref: [docs/PLAN.md](docs/PLAN.md)"

gh issue create --repo "$REPO" \
  --title "HuggingFace Hub Inference Backend" \
  --label "inference,feature" \
  --body "## Summary
Flexible HF backend supporting serverless API, dedicated endpoints, and local transformers.

## Tasks
- [ ] \`HFBackend\` with three modes: \`api\`, \`endpoint\`, \`local\`
- [ ] \`api\` mode: \`huggingface_hub.AsyncInferenceClient\`
- [ ] \`endpoint\` mode: HF Dedicated Endpoint chat completion
- [ ] \`local\` mode: transformers pipeline in thread executor
- [ ] Unified interface matching \`InferenceBackend\`
- [ ] Unit tests for each mode

Ref: [docs/PLAN.md](docs/PLAN.md)"

gh issue create --repo "$REPO" \
  --title "Multi-Model Orchestrator" \
  --label "inference,feature" \
  --body "## Summary
Orchestrate different backends for generator, judge, and rewriter roles with retry logic.

## Tasks
- [ ] \`ModelOrchestrator\` class with \`generator\`, \`judge\`, \`rewriter\` properties
- [ ] \`_build_backend(cfg)\` factory function
- [ ] \`generate_with_retry()\` with exponential backoff
- [ ] Optional \`validate_fn\` callback per generation
- [ ] Unit tests for retry logic

Ref: [docs/PLAN.md](docs/PLAN.md)"

# ─── Phase 3: Data Generation Pipelines ──────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "SFT Data Generation Pipeline" \
  --label "pipeline,feature" \
  --body "## Summary
Generate supervised fine-tuning datasets from seed topics or instructions.

## Tasks
- [ ] \`SFTPipeline\` class using \`ModelOrchestrator\`
- [ ] Load seeds from JSONL/CSV file or inline YAML list
- [ ] Multi-turn conversation generation (user/assistant)
- [ ] Output schemas: \`AlpacaRecord\`, \`ShareGPTRecord\`
- [ ] Support Alpaca, ShareGPT, OpenAI Chat output formats
- [ ] Pandas DataFrame orchestration + JSONL writer
- [ ] Example config: \`configs/sft.yaml\`
- [ ] Integration test generating 10 samples with a mock backend

Ref: [docs/PLAN.md](docs/PLAN.md)"

gh issue create --repo "$REPO" \
  --title "DPO / Preference Data Generation Pipeline" \
  --label "pipeline,feature" \
  --body "## Summary
Generate preference datasets (chosen/rejected pairs) for DPO training.

## Tasks
- [ ] \`DPOPipeline\` class
- [ ] Best-of-N sampling: generate N candidates per prompt at varied temperatures
- [ ] \`LLMJudge\` scoring to select chosen/rejected pair
- [ ] \`DPORecord\` schema (\`prompt\`, \`chosen\`, \`rejected\`)
- [ ] Configurable N and judge model
- [ ] Example config: \`configs/dpo.yaml\`
- [ ] Integration test

Ref: [docs/PLAN.md](docs/PLAN.md)"

gh issue create --repo "$REPO" \
  --title "LLM Evaluation Dataset Generation Pipeline" \
  --label "pipeline,feature" \
  --body "## Summary
Generate evaluation benchmarks: MCQA and open-ended with reference answers.

## Tasks
- [ ] \`EvalPipeline\` class
- [ ] MCQA generation with distractor synthesis — \`MCQARecord\`
- [ ] Open-ended generation (reasoning, coding, math) — \`EvalRecord\`
- [ ] Reference answer + rubric generation
- [ ] \`to_lm_eval_dict()\` output compatible with \`lm-evaluation-harness\`
- [ ] Example config: \`configs/eval.yaml\`
- [ ] Integration test

Ref: [docs/PLAN.md](docs/PLAN.md)"

gh issue create --repo "$REPO" \
  --title "Arena-Style Comparison Data Generation Pipeline" \
  --label "pipeline,feature" \
  --body "## Summary
Generate multi-model pairwise comparison data for arena-style evaluation.

## Tasks
- [ ] \`ArenaPipeline\` class
- [ ] Collect responses from N models for the same prompt
- [ ] LLM judge pairwise comparison (win/loss/tie + reasoning)
- [ ] ELO score tracking across samples
- [ ] \`ArenaRecord\` + \`ModelResponse\` schemas
- [ ] Output format compatible with Chatbot Arena / FastChat
- [ ] Example config: \`configs/arena.yaml\`

Ref: [docs/PLAN.md](docs/PLAN.md)"

gh issue create --repo "$REPO" \
  --title "Multimodal Data Generation Pipeline" \
  --label "pipeline,feature,multimodal" \
  --body "## Summary
Generate image+text instruction pairs for multimodal model finetuning.

## Tasks
- [ ] \`MultimodalPipeline\` class
- [ ] Image captioning and VQA via vision-language models (LLaVA, InternVL)
- [ ] Text-to-image prompt generation for diffusion model finetuning
- [ ] \`MultimodalRecord\` schema (LLaVA-style)
- [ ] Image source: local directory or HF dataset
- [ ] Example config: \`configs/multimodal.yaml\`

Ref: [docs/PLAN.md](docs/PLAN.md)"

# ─── Phase 4: Data Quality & Enhancement ─────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "Adaption SDK Integration for Data Enhancement" \
  --label "quality,feature" \
  --body "## Summary
Post-processing enhancement step using the Adaption SDK.

## Tasks
- [ ] \`AdaptionEnhancer\` class wrapping the Adaption SDK client
- [ ] Quality scoring: score each record and attach \`QualityScore\`
- [ ] Filtering: drop records below \`min_quality_score\`
- [ ] Rewriting: optionally rewrite low-quality records
- [ ] Configurable pipeline: score → filter → rewrite
- [ ] Unit tests with mocked Adaption API responses

Ref: [docs/PLAN.md](docs/PLAN.md)"

gh issue create --repo "$REPO" \
  --title "LLM-as-Judge Quality Scorer" \
  --label "quality,feature" \
  --body "## Summary
Standalone LLM judge for scoring any generated dataset on multiple criteria.

## Tasks
- [ ] \`LLMJudge\` class using \`ModelOrchestrator\`
- [ ] Score criteria: helpfulness, accuracy, safety, instruction-following
- [ ] Structured JSON output from judge → \`QualityScore\`
- [ ] Batch scoring with pandas DataFrame I/O
- [ ] Configurable rubric via Jinja2 template
- [ ] Unit tests with mocked judge responses

Ref: [docs/PLAN.md](docs/PLAN.md)"

gh issue create --repo "$REPO" \
  --title "Deduplication & Filtering Pipeline" \
  --label "quality,feature" \
  --body "## Summary
Remove near-duplicates and low-quality records from generated datasets.

## Tasks
- [ ] MinHash near-duplicate detection (\`datasketch\`)
- [ ] Rule-based filters: min/max length, language detection (\`langdetect\`), regex blocklist
- [ ] Embedding-based semantic dedup (\`sentence-transformers\` + cosine similarity)
- [ ] Pandas-native API: \`deduplicate(df)\`, \`apply_filters(df, rules)\`
- [ ] Configurable via pipeline config (\`dedup_threshold\`, \`dedup_method\`)
- [ ] Unit tests

Ref: [docs/PLAN.md](docs/PLAN.md)"

gh issue create --repo "$REPO" \
  --title "Prompt Template Manager" \
  --label "infrastructure,feature" \
  --body "## Summary
Jinja2-based prompt template system with a registry of built-in templates.

## Tasks
- [ ] \`TemplateRegistry\` class
- [ ] Built-in templates for SFT, DPO, eval judge, arena judge, multimodal
- [ ] Load custom templates from file path in config
- [ ] Template versioning (name + version key in registry)
- [ ] Unit tests for rendering

Ref: [docs/PLAN.md](docs/PLAN.md)"

# ─── Phase 5: Output & Distribution ──────────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "HuggingFace Datasets Exporter" \
  --label "exporter,feature" \
  --body "## Summary
Export generated datasets directly to HuggingFace Hub.

## Tasks
- [ ] \`HFExporter\` class using \`datasets\` library
- [ ] Auto-generate dataset card (README.md) from pipeline metadata
- [ ] Train/val/test splitting with configurable ratios
- [ ] Streaming upload for large datasets (\`push_to_hub\`)
- [ ] Create repo if it doesn't exist
- [ ] Unit tests with mocked HF API

Ref: [docs/PLAN.md](docs/PLAN.md)"

gh issue create --repo "$REPO" \
  --title "Kaggle Datasets Exporter" \
  --label "exporter,feature" \
  --body "## Summary
Export generated datasets to Kaggle in Parquet format.

## Tasks
- [ ] \`KaggleExporter\` class using \`kaggle\` Python package
- [ ] Export to Parquet + \`dataset-metadata.json\`
- [ ] Create new dataset or update existing version
- [ ] Auto-generate Kaggle metadata from pipeline config
- [ ] Unit tests with mocked Kaggle API

Ref: [docs/PLAN.md](docs/PLAN.md)"

# ─── Phase 6: Scale & Performance ────────────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "Distributed Generation with Ray" \
  --label "scale,feature" \
  --body "## Summary
Ray-based distributed pipeline for large-scale generation across multiple workers.

## Tasks
- [ ] \`DistributedPipeline\` wrapper using Ray actors
- [ ] Partition seed data across \`num_workers\` workers
- [ ] Centralized result aggregation with pandas
- [ ] Progress tracking via Rich progress bar
- [ ] Checkpoint/resume: skip seeds already processed
- [ ] Integration test with Ray in local mode

Ref: [docs/PLAN.md](docs/PLAN.md)"

gh issue create --repo "$REPO" \
  --title "Caching & Checkpointing" \
  --label "scale,feature" \
  --body "## Summary
Cache LLM responses and checkpoint pipeline state to support resumable runs.

## Tasks
- [ ] \`CacheManager\`: disk cache (\`diskcache\`) or Redis backend
- [ ] Cache key: hash of (model, messages, sampling params)
- [ ] Pipeline-level checkpointing: save DataFrame state after each batch
- [ ] Incremental updates: detect new seeds and only generate for them
- [ ] \`checkpoint_every\` config option
- [ ] Unit tests

Ref: [docs/PLAN.md](docs/PLAN.md)"

# ─── Phase 7: Tooling & DX ────────────────────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "CLI Interface" \
  --label "dx,feature" \
  --body "## Summary
A \`click\`-based CLI so the full pipeline can be run from the command line.

## Tasks
- [ ] Entry point: \`adaption-datagen\`
- [ ] \`generate --config <path>\` sub-command
- [ ] \`enhance --input <path> --output <path>\` sub-command
- [ ] \`export hf --dataset <path> --repo-id <id>\` sub-command
- [ ] \`export kaggle --dataset <path> --dataset-id <id>\` sub-command
- [ ] \`evaluate --dataset <path>\` sub-command
- [ ] Rich progress bars and summary tables
- [ ] Unit tests for CLI commands

Ref: [docs/PLAN.md](docs/PLAN.md)"

gh issue create --repo "$REPO" \
  --title "Example Configs & Notebooks" \
  --label "dx,documentation" \
  --body "## Summary
Provide ready-to-run configs and notebooks demonstrating the full pipeline.

## Tasks
- [ ] \`configs/sft.yaml\`
- [ ] \`configs/dpo.yaml\`
- [ ] \`configs/eval.yaml\`
- [ ] \`configs/arena.yaml\`
- [ ] \`configs/multimodal.yaml\`
- [ ] \`notebooks/01_sft_quickstart.ipynb\`
- [ ] \`notebooks/02_dpo_preference.ipynb\`
- [ ] \`notebooks/03_eval_generation.ipynb\`
- [ ] \`notebooks/04_multimodal.ipynb\`
- [ ] \`notebooks/05_scale_with_ray.ipynb\`
- [ ] Sample seed files in \`data/seeds/\`

Ref: [docs/PLAN.md](docs/PLAN.md)"

gh issue create --repo "$REPO" \
  --title "CI/CD Pipeline (GitHub Actions)" \
  --label "infrastructure,ci" \
  --body "## Summary
GitHub Actions workflows for linting, type-checking, testing, and publishing.

## Tasks
- [ ] \`.github/workflows/ci.yml\`: ruff lint → mypy type-check → pytest (no GPU)
- [ ] \`.github/workflows/publish.yml\`: build + publish to PyPI on tagged release
- [ ] Badge in README for CI status
- [ ] Coverage report upload to Codecov

Ref: [docs/PLAN.md](docs/PLAN.md)"

gh issue create --repo "$REPO" \
  --title "Comprehensive Documentation" \
  --label "documentation" \
  --body "## Summary
Full documentation covering architecture, quickstart, API reference, and contribution guide.

## Tasks
- [ ] Update \`README.md\`: architecture diagram, quickstart, supported formats table
- [ ] \`docs/\` site with MkDocs Material theme
- [ ] API reference auto-generated with \`mkdocstrings[python]\`
- [ ] \`CONTRIBUTING.md\`: dev setup, coding style, PR process
- [ ] \`mkdocs.yml\` config

Ref: [docs/PLAN.md](docs/PLAN.md)"

echo ""
echo "✅ All 23 issues created in $REPO"
