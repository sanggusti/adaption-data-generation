# Adaption Data Generation

A flexible, config-driven dataset generation framework that sources data from HuggingFace Hub, Kaggle, or local files and generates responses using HuggingFace Transformers, vLLM, Unsloth, or the Adaption API.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run with defaults (HuggingFace dataset + HF Transformers model)
python main.py

# Override dataset and model on the fly
python main.py dataset=kaggle model=vllm generation=adaption experiment_name=my_run

# Run tests
pytest tests/ -v
```

## Project Structure

```
configs/          Hydra YAML configs
  dataset/        huggingface.yaml | kaggle.yaml | local.yaml
  model/          hf_transformers.yaml | vllm.yaml | unsloth.yaml
  generation/     default.yaml | adaption.yaml
models/           Model inference backends
src/
  data/           Loaders + processors
  generation/     DataGenerator pipeline
  utils/          Helpers (seed, logging)
tests/            pytest test suite
docs/             Figures and documentation
main.py           Entry point
```

## Configuration

All settings live in `configs/`. Hydra lets you compose and override them:

```bash
# Use a local CSV, run vLLM inference, save as CSV
python main.py dataset=local model=vllm generation=default \
    dataset.path=data/my_data.csv generation.output_format=csv
```

## Dataset Sources

| Type          | Config key     | Requirements                         |
|---------------|----------------|--------------------------------------|
| HuggingFace   | `huggingface`  | `datasets` package                   |
| Kaggle        | `kaggle`       | `KAGGLE_USERNAME` + `KAGGLE_KEY` env |
| Local file    | `local`        | CSV / JSON / JSONL / Parquet         |

## Model Backends

| Key               | Backend                    | Extra install          |
|-------------------|----------------------------|------------------------|
| `hf_transformers` | HuggingFace Transformers   | *(included)*           |
| `vllm`            | vLLM                       | `pip install -e.[vllm]`|
| `unsloth`         | Unsloth                    | `pip install -e.[unsloth]` |

## Adaption API

Set environment variables and use `generation=adaption`:

```bash
export ADAPTION_API_KEY=your_key_here
python main.py
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check .
```

## License

MIT
