#!/usr/bin/env node
/**
 * create_issues.js
 * Creates all 23 project issues using the GitHub Octokit REST client.
 *
 * Prerequisites:
 *   npm install @octokit/rest
 *
 * Usage:
 *   GITHUB_TOKEN=ghp_your_token node scripts/create_issues.js
 *
 * Optional: set GITHUB_REPO_OWNER and GITHUB_REPO_NAME env vars
 * (defaults to sanggusti / adaption-data-generation).
 */

const { Octokit } = require("@octokit/rest");

const OWNER = process.env.GITHUB_REPO_OWNER || "sanggusti";
const REPO  = process.env.GITHUB_REPO_NAME  || "adaption-data-generation";
const TOKEN = process.env.GITHUB_TOKEN;

if (!TOKEN) {
  console.error("❌  Set the GITHUB_TOKEN environment variable before running.");
  process.exit(1);
}

const octokit = new Octokit({ auth: TOKEN });

// ─── Labels to create (idempotent) ──────────────────────────────────────────

const LABELS = [
  { name: "infrastructure", color: "0075ca", description: "Project infrastructure and setup" },
  { name: "inference",      color: "e4e669", description: "Inference backends" },
  { name: "pipeline",       color: "d93f0b", description: "Data generation pipelines" },
  { name: "quality",        color: "0e8a16", description: "Data quality and enhancement" },
  { name: "exporter",       color: "5319e7", description: "Dataset exporters" },
  { name: "scale",          color: "b60205", description: "Scale and performance" },
  { name: "dx",             color: "c5def5", description: "Developer experience" },
  { name: "documentation",  color: "bfd4f2", description: "Documentation" },
  { name: "multimodal",     color: "fef2c0", description: "Multimodal features" },
  { name: "feature",        color: "a2eeef", description: "New feature" },
  { name: "ci",             color: "1d76db", description: "CI/CD" },
  { name: "good first issue", color: "7057ff", description: "Good for newcomers" },
];

// ─── Issue definitions ───────────────────────────────────────────────────────

const ISSUES = [
  // ── Phase 1: Foundation ────────────────────────────────────────────────────
  {
    title: "Project Scaffolding & Directory Structure",
    labels: ["infrastructure", "good first issue"],
    body: `## Summary
Set up the clean project layout for the package.

## Directory structure
\`\`\`
src/adaption_datagen/
  schemas/       # Pydantic data models
  inference/     # vLLM and HF Hub backends
  pipelines/     # data generation pipelines
  enhancers/     # Adaption SDK integration
  exporters/     # HF Datasets, Kaggle exporters
  utils/         # shared utilities
configs/         # YAML configs per use-case
data/            # local staging area
scripts/         # helper scripts
tests/unit/
tests/integration/
notebooks/
docs/
\`\`\`

## Tasks
- [ ] Create all directories with \`__init__.py\` stubs
- [ ] Add \`src/adaption_datagen/_version.py\`
- [ ] Add top-level \`LICENSE\` (MIT)

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
  {
    title: "Dependency & Environment Setup",
    labels: ["infrastructure"],
    body: `## Summary
Set up package metadata, dependency management, and containerisation.

## Tasks
- [ ] \`pyproject.toml\` with hatchling build backend
- [ ] Core deps: \`vllm\`, \`transformers\`, \`datasets\`, \`pandas\`, \`pydantic\`, \`pydantic-settings\`, \`kaggle\`, \`openai\`, \`huggingface-hub\`, \`ray[default]\`, \`diskcache\`, \`datasketch\`, \`jinja2\`, \`click\`, \`rich\`
- [ ] Optional extras: \`[vllm]\`, \`[dev]\`, \`[docs]\`
- [ ] Dev deps: \`pytest\`, \`pytest-asyncio\`, \`ruff\`, \`mypy\`
- [ ] \`Dockerfile\` (CUDA base image for vLLM)
- [ ] \`docker-compose.yml\` (vLLM server + pipeline runner)
- [ ] \`.env.example\` with all required env vars

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
  {
    title: "Configuration Management (Pydantic + YAML)",
    labels: ["infrastructure"],
    body: `## Summary
Centralised, validated configuration with YAML input and env-var overrides.

## Tasks
- [ ] Pydantic models: \`PipelineConfig\`, \`ModelConfig\`, \`GenerationConfig\`, \`EnhancementConfig\`, \`OutputConfig\`, \`ExportConfig\`, \`CacheConfig\`, \`DistributedConfig\`
- [ ] \`MultiModelConfig\`: generator / judge / rewriter roles
- [ ] \`Settings\` class via \`pydantic-settings\` for env overrides
- [ ] \`load_config(path)\` utility
- [ ] Unit tests for validation and env override

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },

  // ── Phase 2: Inference ─────────────────────────────────────────────────────
  {
    title: "vLLM / OpenAI-compatible Inference Backend",
    labels: ["inference", "feature"],
    body: `## Summary
Async inference backend for vLLM, OpenAI, and any OpenAI-compatible server.

## Tasks
- [ ] \`InferenceBackend\` abstract base class (\`generate\`, \`generate_batch\`)
- [ ] \`GenerationRequest\` and \`GenerationResult\` dataclasses
- [ ] \`OpenAIBackend\` using \`openai.AsyncOpenAI\`
- [ ] Bounded concurrency via \`asyncio.Semaphore\`
- [ ] Unit tests with mocked responses

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
  {
    title: "HuggingFace Hub Inference Backend",
    labels: ["inference", "feature"],
    body: `## Summary
Flexible HF backend: serverless API, dedicated endpoints, and local transformers.

## Tasks
- [ ] \`HFBackend\` with modes: \`api\`, \`endpoint\`, \`local\`
- [ ] \`api\`: \`huggingface_hub.AsyncInferenceClient\`
- [ ] \`endpoint\`: HF Dedicated Endpoint chat completion
- [ ] \`local\`: transformers pipeline in thread executor
- [ ] Unified interface matching \`InferenceBackend\`
- [ ] Unit tests for each mode

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
  {
    title: "Multi-Model Orchestrator",
    labels: ["inference", "feature"],
    body: `## Summary
Orchestrate different backends for generator, judge, and rewriter roles.

## Tasks
- [ ] \`ModelOrchestrator\` with \`generator\`, \`judge\`, \`rewriter\` properties
- [ ] \`_build_backend(cfg)\` factory function
- [ ] \`generate_with_retry()\` with exponential backoff
- [ ] Optional \`validate_fn\` callback
- [ ] Unit tests for retry logic

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },

  // ── Phase 3: Pipelines ────────────────────────────────────────────────────
  {
    title: "SFT Data Generation Pipeline",
    labels: ["pipeline", "feature"],
    body: `## Summary
Generate supervised fine-tuning datasets from seed topics or instructions.

## Tasks
- [ ] \`SFTPipeline\` class
- [ ] Load seeds from JSONL/CSV or inline YAML list
- [ ] Multi-turn conversation generation
- [ ] Output schemas: \`AlpacaRecord\`, \`ShareGPTRecord\`
- [ ] Support Alpaca, ShareGPT, OpenAI Chat formats
- [ ] Example config: \`configs/sft.yaml\`
- [ ] Integration test (10 samples, mock backend)

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
  {
    title: "DPO / Preference Data Generation Pipeline",
    labels: ["pipeline", "feature"],
    body: `## Summary
Generate preference datasets (chosen/rejected pairs) for DPO training.

## Tasks
- [ ] \`DPOPipeline\` class
- [ ] Best-of-N sampling at varied temperatures
- [ ] LLM judge scoring to select chosen/rejected
- [ ] \`DPORecord\` schema
- [ ] Example config: \`configs/dpo.yaml\`

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
  {
    title: "LLM Evaluation Dataset Generation Pipeline",
    labels: ["pipeline", "feature"],
    body: `## Summary
Generate evaluation benchmarks: MCQA and open-ended with reference answers.

## Tasks
- [ ] \`EvalPipeline\` class
- [ ] MCQA generation with distractors — \`MCQARecord\`
- [ ] Open-ended generation — \`EvalRecord\`
- [ ] Reference answer + rubric generation
- [ ] \`to_lm_eval_dict()\` output for \`lm-evaluation-harness\`
- [ ] Example config: \`configs/eval.yaml\`

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
  {
    title: "Arena-Style Comparison Data Generation Pipeline",
    labels: ["pipeline", "feature"],
    body: `## Summary
Generate multi-model pairwise comparison data for arena-style evaluation.

## Tasks
- [ ] \`ArenaPipeline\` class
- [ ] Responses from N models per prompt
- [ ] LLM judge pairwise comparison (win/loss/tie)
- [ ] ELO score tracking
- [ ] \`ArenaRecord\` + \`ModelResponse\` schemas
- [ ] Example config: \`configs/arena.yaml\`

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
  {
    title: "Multimodal Data Generation Pipeline",
    labels: ["pipeline", "feature", "multimodal"],
    body: `## Summary
Generate image+text instruction pairs for multimodal model finetuning.

## Tasks
- [ ] \`MultimodalPipeline\` class
- [ ] Image captioning and VQA via VLMs
- [ ] Text-to-image prompt generation
- [ ] \`MultimodalRecord\` schema (LLaVA-style)
- [ ] Image source: local dir or HF dataset
- [ ] Example config: \`configs/multimodal.yaml\`

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },

  // ── Phase 4: Quality ──────────────────────────────────────────────────────
  {
    title: "Adaption SDK Integration for Data Enhancement",
    labels: ["quality", "feature"],
    body: `## Summary
Post-processing enhancement using the Adaption SDK.

## Tasks
- [ ] \`AdaptionEnhancer\` wrapping Adaption SDK client
- [ ] Quality scoring + attach \`QualityScore\` to records
- [ ] Filter records below \`min_quality_score\`
- [ ] Optional rewriting of low-quality records
- [ ] Configurable pipeline: score → filter → rewrite
- [ ] Unit tests with mocked Adaption API

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
  {
    title: "LLM-as-Judge Quality Scorer",
    labels: ["quality", "feature"],
    body: `## Summary
Standalone LLM judge for scoring any dataset on multiple criteria.

## Tasks
- [ ] \`LLMJudge\` class
- [ ] Criteria: helpfulness, accuracy, safety, instruction-following
- [ ] Structured JSON → \`QualityScore\`
- [ ] Batch scoring with pandas DataFrame I/O
- [ ] Configurable rubric via Jinja2 template

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
  {
    title: "Deduplication & Filtering Pipeline",
    labels: ["quality", "feature"],
    body: `## Summary
Remove near-duplicates and low-quality records from generated datasets.

## Tasks
- [ ] MinHash dedup (\`datasketch\`)
- [ ] Rule-based filters: length, language, regex blocklist
- [ ] Embedding-based semantic dedup (\`sentence-transformers\`)
- [ ] Pandas API: \`deduplicate(df)\`, \`apply_filters(df, rules)\`

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
  {
    title: "Prompt Template Manager",
    labels: ["infrastructure", "feature"],
    body: `## Summary
Jinja2-based prompt template system with a registry of built-in templates.

## Tasks
- [ ] \`TemplateRegistry\` class
- [ ] Built-in templates for SFT, DPO, eval, arena, multimodal
- [ ] Load custom templates from file in config
- [ ] Template versioning
- [ ] Unit tests for rendering

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },

  // ── Phase 5: Export ───────────────────────────────────────────────────────
  {
    title: "HuggingFace Datasets Exporter",
    labels: ["exporter", "feature"],
    body: `## Summary
Export generated datasets directly to HuggingFace Hub.

## Tasks
- [ ] \`HFExporter\` using \`datasets\` library
- [ ] Auto-generate dataset card from pipeline metadata
- [ ] Train/val/test splitting
- [ ] Streaming upload (\`push_to_hub\`)
- [ ] Create repo if not exists

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
  {
    title: "Kaggle Datasets Exporter",
    labels: ["exporter", "feature"],
    body: `## Summary
Export generated datasets to Kaggle in Parquet format.

## Tasks
- [ ] \`KaggleExporter\` using \`kaggle\` package
- [ ] Export Parquet + \`dataset-metadata.json\`
- [ ] Create or update dataset version
- [ ] Auto-generate Kaggle metadata from config

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },

  // ── Phase 6: Scale ────────────────────────────────────────────────────────
  {
    title: "Distributed Generation with Ray",
    labels: ["scale", "feature"],
    body: `## Summary
Ray-based distributed pipeline for large-scale generation.

## Tasks
- [ ] \`DistributedPipeline\` wrapper using Ray actors
- [ ] Partition seeds across \`num_workers\`
- [ ] Centralized result aggregation with pandas
- [ ] Progress via Rich progress bar
- [ ] Checkpoint / resume support
- [ ] Integration test (Ray local mode)

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
  {
    title: "Caching & Checkpointing",
    labels: ["scale", "feature"],
    body: `## Summary
Cache LLM responses and checkpoint pipeline state for resumable runs.

## Tasks
- [ ] \`CacheManager\`: disk (\`diskcache\`) or Redis backend
- [ ] Cache key: hash(model, messages, sampling params)
- [ ] Pipeline checkpoint after each batch
- [ ] Incremental updates for new seeds only
- [ ] \`checkpoint_every\` config option

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },

  // ── Phase 7: DX ───────────────────────────────────────────────────────────
  {
    title: "CLI Interface",
    labels: ["dx", "feature"],
    body: `## Summary
Click-based CLI so the full pipeline can be run from the command line.

## Tasks
- [ ] Entry point: \`adaption-datagen\`
- [ ] \`generate --config <path>\`
- [ ] \`enhance --input <path> --output <path>\`
- [ ] \`export hf\` / \`export kaggle\`
- [ ] \`evaluate --dataset <path>\`
- [ ] Rich progress bars and summary tables

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
  {
    title: "Example Configs & Notebooks",
    labels: ["dx", "documentation"],
    body: `## Summary
Ready-to-run configs and notebooks demonstrating the full pipeline.

## Tasks
- [ ] \`configs/sft.yaml\`, \`configs/dpo.yaml\`, \`configs/eval.yaml\`, \`configs/arena.yaml\`, \`configs/multimodal.yaml\`
- [ ] \`notebooks/01_sft_quickstart.ipynb\`
- [ ] \`notebooks/02_dpo_preference.ipynb\`
- [ ] \`notebooks/03_eval_generation.ipynb\`
- [ ] \`notebooks/04_multimodal.ipynb\`
- [ ] \`notebooks/05_scale_with_ray.ipynb\`
- [ ] Sample seeds in \`data/seeds/\`

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
  {
    title: "CI/CD Pipeline (GitHub Actions)",
    labels: ["infrastructure", "ci"],
    body: `## Summary
GitHub Actions workflows for linting, type-checking, testing, and publishing.

## Tasks
- [ ] \`.github/workflows/ci.yml\`: ruff → mypy → pytest
- [ ] \`.github/workflows/publish.yml\`: build + PyPI publish on tag
- [ ] Coverage upload to Codecov
- [ ] CI badge in README

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
  {
    title: "Comprehensive Documentation",
    labels: ["documentation"],
    body: `## Summary
Full docs: architecture, quickstart, API reference, contribution guide.

## Tasks
- [ ] Update \`README.md\` with architecture diagram, quickstart, formats table
- [ ] MkDocs Material site under \`docs/\`
- [ ] API reference with \`mkdocstrings[python]\`
- [ ] \`CONTRIBUTING.md\`
- [ ] \`mkdocs.yml\` config

Ref: [docs/PLAN.md](docs/PLAN.md)`,
  },
];

// ─── Milestones ──────────────────────────────────────────────────────────────

const MILESTONES = [
  { title: "v0.1 — Foundation",  description: "Installable package with CI",               due_on: null },
  { title: "v0.2 — Inference",   description: "Both backends working",                      due_on: null },
  { title: "v0.3 — Pipelines",   description: "SFT + DPO generation end-to-end",            due_on: null },
  { title: "v0.4 — Quality",     description: "Enhancement + dedup working",                due_on: null },
  { title: "v0.5 — Export",      description: "One-click HF Hub push + CLI",                due_on: null },
  { title: "v0.6 — Full",        description: "Arena, multimodal, Ray scale, full docs",    due_on: null },
];

// Issue index → milestone title mapping (0-indexed)
const ISSUE_MILESTONE = {
  0:  "v0.1 — Foundation",   // scaffolding
  1:  "v0.1 — Foundation",   // deps
  2:  "v0.1 — Foundation",   // config
  21: "v0.1 — Foundation",   // CI/CD
  3:  "v0.2 — Inference",
  4:  "v0.2 — Inference",
  5:  "v0.2 — Inference",
  6:  "v0.3 — Pipelines",    // SFT
  7:  "v0.3 — Pipelines",    // DPO
  8:  "v0.3 — Pipelines",    // eval
  14: "v0.3 — Pipelines",    // templates
  11: "v0.4 — Quality",      // adaption
  12: "v0.4 — Quality",      // judge
  13: "v0.4 — Quality",      // dedup
  15: "v0.5 — Export",       // HF
  16: "v0.5 — Export",       // kaggle
  19: "v0.5 — Export",       // CLI
  9:  "v0.6 — Full",         // arena
  10: "v0.6 — Full",         // multimodal
  17: "v0.6 — Full",         // ray
  18: "v0.6 — Full",         // cache
  20: "v0.6 — Full",         // notebooks
  22: "v0.6 — Full",         // docs
};

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  console.log(`\nCreating labels in ${OWNER}/${REPO} …`);
  for (const label of LABELS) {
    try {
      await octokit.issues.createLabel({ owner: OWNER, repo: REPO, ...label });
      process.stdout.write(`  ✅ label: ${label.name}\n`);
    } catch (err) {
      // 422 = already exists
      if (err.status === 422) {
        process.stdout.write(`  ⏭  label exists: ${label.name}\n`);
      } else {
        console.error(`  ❌ label ${label.name}: ${err.message}`);
      }
    }
  }

  console.log(`\nCreating milestones in ${OWNER}/${REPO} …`);
  const milestoneNumbers = {};
  for (const ms of MILESTONES) {
    try {
      const { data } = await octokit.issues.createMilestone({
        owner: OWNER, repo: REPO, title: ms.title, description: ms.description,
      });
      milestoneNumbers[ms.title] = data.number;
      process.stdout.write(`  ✅ milestone: ${ms.title} (#${data.number})\n`);
    } catch (err) {
      if (err.status === 422) {
        // Already exists — fetch it
        const { data: list } = await octokit.issues.listMilestones({ owner: OWNER, repo: REPO });
        const existing = list.find(m => m.title === ms.title);
        if (existing) {
          milestoneNumbers[ms.title] = existing.number;
          process.stdout.write(`  ⏭  milestone exists: ${ms.title} (#${existing.number})\n`);
        }
      } else {
        console.error(`  ❌ milestone ${ms.title}: ${err.message}`);
      }
    }
  }

  console.log(`\nCreating ${ISSUES.length} issues in ${OWNER}/${REPO} …`);
  for (let i = 0; i < ISSUES.length; i++) {
    const issue = ISSUES[i];
    const mstitle = ISSUE_MILESTONE[i];
    const milestone = mstitle ? milestoneNumbers[mstitle] : undefined;
    try {
      const { data } = await octokit.issues.create({
        owner: OWNER,
        repo: REPO,
        title: issue.title,
        body: issue.body,
        labels: issue.labels,
        milestone,
      });
      console.log(`  ✅ #${data.number}: ${issue.title}`);
    } catch (err) {
      console.error(`  ❌ issue "${issue.title}": ${err.message}`);
    }
  }

  console.log("\n✅ Done.");
}

main().catch(err => { console.error(err); process.exit(1); });
