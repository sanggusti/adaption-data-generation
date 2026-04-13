# adaption-data-generation — Overview

## Architecture

```
configs/         Hydra YAML configs (dataset, model, generation)
models/          Model backends: HuggingFace Transformers, vLLM, Unsloth
src/
  data/          Dataset loaders (HF Hub, Kaggle, local) + processors
  generation/    DataGenerator orchestration pipeline
  utils/         Seed setting, logging helpers
tests/           pytest test suite
main.py          Hydra entry point
```

## Pipeline

1. **Load** — `src/data/loaders.py` pulls data from HuggingFace Hub, Kaggle, or a local file.
2. **Process** — `src/data/processors.py` builds prompts and deduplicates rows.
3. **Generate** — `src/generation/generator.py` dispatches to the chosen model backend or Adaption API.
4. **Save** — Output written as JSONL / CSV / Parquet / JSON under `outputs/`.

## Configuration

All configuration is driven by Hydra. Override any field on the command line:

```bash
python main.py dataset=kaggle model=vllm generation=adaption experiment_name=my_run
```

## Model Backends

| Key              | Class                  | Notes                          |
|------------------|------------------------|--------------------------------|
| `hf_transformers`| `HFTransformersModel`  | Standard HF pipeline           |
| `vllm`           | `VLLMModel`            | Fast batched inference (GPU)   |
| `unsloth`        | `UnslothModel`         | 4-bit quantized (Unsloth)      |

## Environment Variables

| Variable                  | Purpose                        |
|---------------------------|--------------------------------|
| `ADAPTION_API_ENDPOINT`   | Adaption API base URL          |
| `ADAPTION_API_KEY`        | Adaption API bearer token      |
| `KAGGLE_USERNAME`         | Kaggle API credentials         |
| `KAGGLE_KEY`              | Kaggle API key                 |
