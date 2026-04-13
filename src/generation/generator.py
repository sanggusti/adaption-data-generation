"""Core data generation pipeline."""

import json
import logging
import os
from typing import List

import pandas as pd
from omegaconf import DictConfig

from models import get_model
from ..data.loaders import load_dataset_from_config
from ..data.processors import build_prompts, deduplicate

logger = logging.getLogger(__name__)


class DataGenerator:
    """Orchestrates dataset loading, prompt building, model inference, and saving."""

    def __init__(self, config: DictConfig):
        self.config = config

    def run(self) -> pd.DataFrame:
        """Execute the full generation pipeline and return a DataFrame."""
        # 1. Load source dataset
        df = load_dataset_from_config(self.config)

        # 2. Optionally limit to num_samples
        num_samples = self.config.generation.get("num_samples")
        if num_samples and len(df) > num_samples:
            df = df.sample(n=num_samples, random_state=self.config.get("seed", 42)).reset_index(drop=True)

        # 3. Deduplicate source
        if self.config.generation.get("deduplicate", True):
            text_col = self.config.dataset.get("text_column", "text")
            df = deduplicate(df, column=text_col if text_col in df.columns else None)

        # 4. Build prompts
        prompts = build_prompts(df, self.config)

        # 5. Run inference
        responses = self._generate_responses(prompts)

        # 6. Assemble output dataframe
        out_df = df.copy()
        out_df["generated_response"] = responses
        out_df["prompt"] = prompts

        # 7. Save results
        self._save(out_df)

        return out_df

    def _generate_responses(self, prompts: List[str]) -> List[str]:
        """Dispatch to the correct generation backend."""
        gen_type = self.config.generation.get("type", "default")

        if gen_type == "adaption":
            return self._call_adaption_api(prompts)

        # Default: use configured model backend
        model = get_model(self.config)
        with model:
            responses = model.generate(prompts)
        return responses

    def _call_adaption_api(self, prompts: List[str]) -> List[str]:
        """Call the Adaption API for generation."""
        import requests

        api_cfg = self.config.generation.adaption_api
        endpoint = api_cfg.get("endpoint", "https://api.adaption.ai")
        api_key = api_cfg.get("api_key", "")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        responses = []
        for i, prompt in enumerate(prompts):
            payload = {
                "model": api_cfg.get("model", "adaption-v1"),
                "prompt": prompt,
                "max_tokens": api_cfg.get("max_tokens", 512),
                "temperature": api_cfg.get("temperature", 0.7),
            }
            try:
                resp = requests.post(
                    f"{endpoint}/v1/completions", headers=headers, json=payload, timeout=60
                )
                resp.raise_for_status()
                responses.append(resp.json()["choices"][0]["text"])
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(
                    f"Adaption API call failed for prompt index {i}: {exc}"
                ) from exc
        return responses

    def _save(self, df: pd.DataFrame) -> None:
        """Persist the generated dataset."""
        output_dir = self.config.get("output_dir", "outputs")
        experiment_name = self.config.get("experiment_name", "experiment")
        os.makedirs(output_dir, exist_ok=True)

        fmt = self.config.generation.get("output_format", "jsonl")
        output_path = os.path.join(output_dir, f"{experiment_name}.{fmt}")

        if fmt == "jsonl":
            df.to_json(output_path, orient="records", lines=True)
        elif fmt == "csv":
            df.to_csv(output_path, index=False)
        elif fmt == "parquet":
            df.to_parquet(output_path, index=False)
        elif fmt == "json":
            df.to_json(output_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported output_format '{fmt}'.")

        logger.info("Saved %d records to %s", len(df), output_path)
