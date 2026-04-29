import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GPU = os.environ.get("MODAL_GPU")
LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_PROJECT_ROOT = Path("/root/project")
REMOTE_OUTPUT_DIR = Path("/data/outputs")

# Resolve requirements
req_file = LOCAL_PROJECT_ROOT / "requirements.txt"
if req_file.exists():
    RUNTIME_REQUIREMENTS = [
        line.strip() for line in req_file.read_text().splitlines() 
        if line.strip() and not line.lstrip().startswith("#")
    ]
else:
    RUNTIME_REQUIREMENTS = [
        "hydra-core>=1.3.2", "omegaconf>=2.3.0", "pandas>=2.0.0", "datasets>=2.18.0",
        "transformers>=4.40.0", "torch>=2.2.0", "kaggle>=1.6.0", "requests>=2.31.0",
        "numpy>=1.26.0", "adaption",
    ]

# Modal Setup
app = modal.App("adaption-data-generation-pipeline")
volume = modal.Volume.from_name("adaption-data", create_if_missing=True)
models_volume = modal.Volume.from_name("adaption-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*RUNTIME_REQUIREMENTS)
    .add_local_file("main.py", remote_path= "/root/project/main.py")
    .add_local_file("requirements.txt", remote_path= "/root/project/requirements.txt")
    .add_local_dir("configs", remote_path= "/root/project/configs")
    .add_local_dir("scripts", remote_path= "/root/project/scripts")
    .add_local_dir("src", remote_path= "/root/project/src")
    .add_local_dir("models", remote_path= "/root/project/models")
)

def get_output_file(output_root: Path, experiment_name: str, fmt: str | None) -> Path:
    if fmt:
        candidate = output_root / f"{experiment_name}.{fmt}"
        if candidate.exists(): return candidate
    else:
        for ext in ("jsonl", "csv", "parquet", "json"):
            candidate = output_root / f"{experiment_name}.{ext}"
            if candidate.exists(): return candidate
            
    raise FileNotFoundError(f"No generated output found for '{experiment_name}' in '{output_root}'.")

@app.function(
    image=image,
    gpu=GPU,
    volumes={"/data": volume, "/models": models_volume},
    secrets=[modal.Secret.from_name("adaption")],
    timeout=86400, # 24 hours
)
def run_generation_and_adapt(
    dataset: str, model: str, generation: str, experiment_name: str,
    output_dir: str = str(REMOTE_OUTPUT_DIR), output_format: str | None = None,
    hallucination_mitigation: bool = False,
):
    os.environ.update({
        "HF_HOME": "/models/huggingface",
        "HF_HUB_CACHE": "/models/huggingface/hub",
        "TRANSFORMERS_CACHE": "/models/huggingface"
    })

    logger.info("=== STEP 1: Running dataset generation ===")
    cmd_args = [
        f"dataset={dataset}", f"model={model}", f"generation={generation}",
        f"experiment_name={experiment_name}", f"output_dir={output_dir}"
    ]
    if output_format:
        cmd_args.append(f"generation.output_format={output_format}")

    subprocess.run([sys.executable, "main.py", *cmd_args], check=True, cwd=REMOTE_PROJECT_ROOT)

    logger.info("=== STEP 2: Locating generated dataset ===")
    output_file = get_output_file(Path(output_dir), experiment_name, output_format)
    logger.info(f"Found output file: {output_file}")

    logger.info("=== STEP 3: Enhancing dataset using Adaption API ===")
    from adaption import Adaption, DatasetTimeout
    client = Adaption()

    dataset_id = client.datasets.upload_file(str(output_file)).dataset_id
    logger.info(f"Upload complete. Dataset ID: {dataset_id}. Waiting for readiness...")

    while client.datasets.get_status(dataset_id).row_count is None:
        time.sleep(2)

    run_kwargs = {"column_mapping": {"prompt": "prompt", "completion": "generated_response"}}
    if hallucination_mitigation:
        run_kwargs["brand_controls"] = {"hallucination_mitigation": True}

    run = client.datasets.run(dataset_id, **run_kwargs)
    logger.info(f"Adaption run started! ID: {run.run_id},~{run.estimated_minutes} min")
    logger.info(f"Run details: {run.estimated_credits_consumed} credits")

    try:
        final = client.datasets.wait_for_completion(dataset_id, timeout=1800)
        if final.error: raise RuntimeError(final.error.message)
    except DatasetTimeout as exc:
        logger.warning(f"Pipeline timed out after {exc.timeout}s.")
        return {"status": "timed_out", "dataset_id": dataset_id, "generated_output": str(output_file)}

    download_url = client.datasets.download(dataset_id)
    # Should be stored in volume after download
    logger.info(f"=== SUCCESS ===\nDownload enhanced dataset at: {download_url}")
    
    return {
        "status": final.status,
        "dataset_id": dataset_id,
        "generated_output": str(output_file),
        "download_url": download_url,
    }

@app.local_entrypoint()
def main(
    dataset: str = "huggingface", model: str = "hf_transformers", generation: str = "default",
    experiment_name: str = "adaption_pipeline_run", output_dir: str = str(REMOTE_OUTPUT_DIR),
    output_format: str | None = None, hallucination_mitigation: bool = False,
):
    """
    Run with: modal run scripts/modal_run_generation.py --experiment-name my_run
    """
    logger.info(f"Triggering Modal task for {dataset} / {model} / {generation}")
    result = run_generation_and_adapt.remote(
        dataset=dataset, model=model, generation=generation, experiment_name=experiment_name,
        output_dir=output_dir, output_format=output_format, 
        hallucination_mitigation=hallucination_mitigation,
    )
    logger.info(f"Remote pipeline result: {result}")
