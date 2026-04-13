"""Dataset loaders for HuggingFace, Kaggle, and local file sources."""

import logging
import os
from typing import Optional

import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_dataset_from_config(config: DictConfig) -> pd.DataFrame:
    """Load a dataset according to its configuration."""
    dataset_type = config.dataset.type
    if dataset_type == "huggingface":
        return _load_huggingface(config.dataset)
    elif dataset_type == "kaggle":
        return _load_kaggle(config.dataset)
    elif dataset_type == "local":
        return _load_local(config.dataset)
    else:
        raise ValueError(
            f"Unknown dataset type '{dataset_type}'. "
            "Choose from: huggingface, kaggle, local"
        )


def _load_huggingface(cfg: DictConfig) -> pd.DataFrame:
    """Load a dataset from HuggingFace Hub."""
    from datasets import load_dataset

    logger.info("Loading HuggingFace dataset: %s (split=%s)", cfg.name, cfg.split)
    ds = load_dataset(
        cfg.name,
        cfg.get("subset"),
        split=cfg.get("split", "train"),
        cache_dir=cfg.get("cache_dir"),
    )
    df = ds.to_pandas()
    if cfg.get("max_samples") and len(df) > cfg.max_samples:
        df = df.sample(n=cfg.max_samples, random_state=42).reset_index(drop=True)
    logger.info("Loaded %d samples from HuggingFace dataset.", len(df))
    return df


def _load_kaggle(cfg: DictConfig) -> pd.DataFrame:
    """Download and load a dataset from Kaggle."""
    import kaggle  # noqa: F401 — requires KAGGLE_USERNAME and KAGGLE_KEY env vars

    download_dir = cfg.get("download_dir", "data/kaggle")
    os.makedirs(download_dir, exist_ok=True)
    dataset_ref = f"{cfg.owner}/{cfg.dataset_name}"
    logger.info("Downloading Kaggle dataset: %s", dataset_ref)
    kaggle.api.dataset_download_files(
        dataset_ref,
        path=download_dir,
        unzip=True,
    )
    files = cfg.get("files") or []
    frames = []
    for fname in files:
        fpath = os.path.join(download_dir, fname)
        logger.info("Reading Kaggle file: %s", fpath)
        frames.append(pd.read_csv(fpath))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if cfg.get("max_samples") and len(df) > cfg.max_samples:
        df = df.sample(n=cfg.max_samples, random_state=42).reset_index(drop=True)
    logger.info("Loaded %d samples from Kaggle dataset.", len(df))
    return df


def _load_local(cfg: DictConfig) -> pd.DataFrame:
    """Load a dataset from a local file."""
    fmt = cfg.get("format", "csv").lower()
    path = cfg.path
    logger.info("Loading local dataset from: %s (format=%s)", path, fmt)
    readers = {
        "csv": pd.read_csv,
        "json": pd.read_json,
        "jsonl": lambda p: pd.read_json(p, lines=True),
        "parquet": pd.read_parquet,
    }
    if fmt not in readers:
        raise ValueError(f"Unsupported local file format '{fmt}'. Choose from: {list(readers)}")
    df = readers[fmt](path)
    if cfg.get("max_samples") and len(df) > cfg.max_samples:
        df = df.sample(n=cfg.max_samples, random_state=42).reset_index(drop=True)
    logger.info("Loaded %d samples from local dataset.", len(df))
    return df
