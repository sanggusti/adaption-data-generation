"""Data processing utilities."""

import logging
from typing import List, Optional

import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_prompts(df: pd.DataFrame, cfg: DictConfig) -> List[str]:
    """Build prompt strings from dataframe rows using the generation template."""
    text_col = cfg.dataset.get("text_column", "text")
    template = cfg.generation.get("prompt_template", "{text}")

    if text_col not in df.columns:
        raise ValueError(
            f"text_column '{text_col}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    prompts = df[text_col].astype(str).apply(lambda t: template.format(text=t)).tolist()
    logger.info("Built %d prompts.", len(prompts))
    return prompts


def deduplicate(df: pd.DataFrame, column: Optional[str] = None) -> pd.DataFrame:
    """Remove duplicate rows (optionally by a specific column)."""
    before = len(df)
    df = df.drop_duplicates(subset=[column] if column else None).reset_index(drop=True)
    logger.info("Deduplicated: %d -> %d rows.", before, len(df))
    return df
