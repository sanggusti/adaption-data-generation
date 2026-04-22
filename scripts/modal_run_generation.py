import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import modal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GPU = os.environ.get("MODAL_GPU")

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_PROJECT_ROOT = Path("/root/project")
REMOTE_OUTPUT_DIR = Path("/data/outputs")
SUPPORTED_OUTPUT_FORMATS = ("jsonl", "csv", "parquet", "json")
DEFAULT_REQUIREMENTS = [
    "hydra-core>=1.3.2",
    "omegaconf>=2.3.0",
    "pandas>=2.0.0",
    "datasets>=2.18.0",
    "transformers>=4.40.0",
    "torch>=2.2.0",
    "kaggle>=1.6.0",
    "requests>=2.31.0",
    "numpy>=1.26.0",
    "adaption",
]


def read_requirements() -> list[str]:
    requirements_file = LOCAL_PROJECT_ROOT / "requirements.txt"
    if not requirements_file.exists():
        return DEFAULT_REQUIREMENTS

    return [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


RUNTIME_REQUIREMENTS = read_requirements()

app = modal.App("adaption-data-generation-pipeline")
volume = modal.Volume.from_name("adaption-data", create_if_missing=True)
models_volume = modal.Volume.from_name("adaption-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(*RUNTIME_REQUIREMENTS)
    .add_local_file(
        str(LOCAL_PROJECT_ROOT / "main.py"),
        remote_path=str(REMOTE_PROJECT_ROOT / "main.py"),
    )
    .add_local_file(
        str(LOCAL_PROJECT_ROOT / "requirements.txt"),
        remote_path=str(REMOTE_PROJECT_ROOT / "requirements.txt"),
    )
    .add_local_dir(
        str(LOCAL_PROJECT_ROOT / "configs"),
        remote_path=str(REMOTE_PROJECT_ROOT / "configs"),
    )
    .add_local_dir(
        str(LOCAL_PROJECT_ROOT / "scripts"),
        remote_path=str(REMOTE_PROJECT_ROOT / "scripts"),
    )
    .add_local_dir(
        str(LOCAL_PROJECT_ROOT / "src"),
        remote_path=str(REMOTE_PROJECT_ROOT / "src"),
    )
    .add_local_dir(
        str(LOCAL_PROJECT_ROOT / "models"),
        remote_path=str(REMOTE_PROJECT_ROOT / "models"),
    )
)


def resolve_output_file(experiment_name: str, output_dir: str, output_format: str | None) -> Path:
    output_root = Path(output_dir)
    if not output_root.exists():
        raise FileNotFoundError(f"Output directory '{output_root}' does not exist.")

    if output_format:
        candidate = output_root / f"{experiment_name}.{output_format}"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Expected output file '{candidate}' was not created.")

    matches = [
        output_root / f"{experiment_name}.{fmt}"
        for fmt in SUPPORTED_OUTPUT_FORMATS
        if (output_root / f"{experiment_name}.{fmt}").exists()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No generated output found for experiment '{experiment_name}' in '{output_root}'."
        )
    if len(matches) > 1:
        joined_matches = ", ".join(str(path) for path in matches)
        raise RuntimeError(
            "Multiple output files matched the experiment name; set generation.output_format "
            f"explicitly. Matches: {joined_matches}"
        )
    return matches[0]


def get_column_mapping(output_file: Path) -> dict[str, str]:
    suffix = output_file.suffix.lower()
    if suffix == ".csv":
        import pandas as pd

        columns = pd.read_csv(output_file, nrows=0).columns.tolist()
    elif suffix == ".jsonl":
        import json

        with output_file.open() as handle:
            first_row = handle.readline().strip()
        if not first_row:
            raise ValueError(f"Generated file '{output_file}' is empty.")
        columns = list(json.loads(first_row).keys())
    elif suffix == ".json":
        import json

        records = json.loads(output_file.read_text())
        if not records:
            raise ValueError(f"Generated file '{output_file}' is empty.")
        if not isinstance(records, list) or not isinstance(records[0], dict):
            raise ValueError(f"Generated JSON file '{output_file}' is not a list of objects.")
        columns = list(records[0].keys())
    elif suffix == ".parquet":
        import pandas as pd

        columns = pd.read_parquet(output_file).columns.tolist()
    else:
        raise ValueError(f"Unsupported output format '{suffix}' for file '{output_file}'.")

    required_columns = {"prompt", "generated_response"}
    missing_columns = required_columns.difference(columns)
    if missing_columns:
        raise ValueError(
            f"Generated file '{output_file}' is missing required columns: {sorted(missing_columns)}"
        )

    return {"prompt": "prompt", "completion": "generated_response"}


def build_hydra_args(
    dataset: str,
    model: str,
    generation: str,
    experiment_name: str,
    output_dir: str,
    output_format: str | None,
) -> list[str]:
    hydra_args = [
        f"dataset={dataset}",
        f"model={model}",
        f"generation={generation}",
        f"experiment_name={experiment_name}",
        f"output_dir={output_dir}",
    ]
    if output_format:
        hydra_args.append(f"generation.output_format={output_format}")
    return hydra_args


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/data": volume, "/models": models_volume},
    secrets=[modal.Secret.from_name("adaption")],
    timeout=3600 * 24,
)
def run_generation_and_adapt(
    dataset: str,
    model: str,
    generation: str,
    experiment_name: str,
    output_dir: str = str(REMOTE_OUTPUT_DIR),
    output_format: str | None = None,
    hallucination_mitigation: bool = False,
):
    os.environ.setdefault("HF_HOME", "/models/huggingface")
    os.environ.setdefault("HF_HUB_CACHE", "/models/huggingface/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/models/huggingface")

    hydra_args = build_hydra_args(
        dataset=dataset,
        model=model,
        generation=generation,
        experiment_name=experiment_name,
        output_dir=output_dir,
        output_format=output_format,
    )

    logger.info("=== STEP 1: Running dataset generation ===")
    cmd = [sys.executable, "main.py", *hydra_args]
    logger.info("Executing: %s", " ".join(cmd))

    subprocess.run(cmd, check=True, cwd=str(REMOTE_PROJECT_ROOT))

    logger.info("=== STEP 2: Locating generated dataset ===")
    output_file = resolve_output_file(
        experiment_name=experiment_name,
        output_dir=output_dir,
        output_format=output_format,
    )
    logger.info("Found newest output file: %s", output_file)

    logger.info("=== STEP 3: Enhancing dataset using Adaption API ===")
    from adaption import Adaption, DatasetTimeout

    client = Adaption()
    column_mapping = cast(Any, get_column_mapping(output_file))

    logger.info("Uploading %s to Adaption...", output_file)
    result = client.datasets.upload_file(str(output_file))
    dataset_id = result.dataset_id
    logger.info("Upload complete. Dataset ID: %s", dataset_id)

    logger.info("Waiting for dataset status to be ready...")
    while True:
        status = client.datasets.get_status(dataset_id)
        if status.row_count is not None:
            break
        time.sleep(2)

    logger.info("Starting Adaption run with column mapping: %s", column_mapping)
    if hallucination_mitigation:
        run = client.datasets.run(
            dataset_id,
            column_mapping=column_mapping,
            brand_controls=cast(Any, {"hallucination_mitigation": True}),
        )
    else:
        run = client.datasets.run(dataset_id, column_mapping=column_mapping)
    logger.info("Adaption run started! ID: %s", run.run_id)

    logger.info("Waiting for completion...")
    try:
        final = client.datasets.wait_for_completion(dataset_id, timeout=1800)
        logger.info("Pipeline finished with status: %s", final.status)
        if final.error:
            raise RuntimeError(final.error.message)
    except DatasetTimeout as exc:
        logger.warning(
            "Adaption pipeline timed out after %ss (last status: %s).",
            exc.timeout,
            exc.last_status,
        )
        return {
            "status": "timed_out",
            "dataset_id": dataset_id,
            "generated_output": str(output_file),
        }

    download_url = client.datasets.download(dataset_id)
    logger.info("=== SUCCESS ===")
    logger.info("Download enhanced dataset at: %s", download_url)
    return {
        "status": final.status,
        "dataset_id": dataset_id,
        "generated_output": str(output_file),
        "download_url": download_url,
    }


@app.local_entrypoint()
def main(
    dataset: str = "huggingface",
    model: str = "hf_transformers",
    generation: str = "default",
    experiment_name: str = "adaption_pipeline_run",
    output_dir: str = str(REMOTE_OUTPUT_DIR),
    output_format: str | None = None,
    hallucination_mitigation: bool = False,
):
    """
    Entry point for the Modal app. Run with:
    modal run scripts/modal_run_generation.py --experiment-name my_run

    Set MODAL_GPU=A10G (or another Modal GPU type) only when the selected model
    backend actually needs accelerator hardware.
    """
    logger.info(
        "Triggering Modal task for dataset=%s model=%s generation=%s output_dir=%s",
        dataset,
        model,
        generation,
        output_dir,
    )
    result = run_generation_and_adapt.remote(
        dataset=dataset,
        model=model,
        generation=generation,
        experiment_name=experiment_name,
        output_dir=output_dir,
        output_format=output_format,
        hallucination_mitigation=hallucination_mitigation,
    )
    logger.info("Remote pipeline result: %s", result)
