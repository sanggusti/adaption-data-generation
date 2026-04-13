"""Tests for dataset loaders."""

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data.loaders import _load_local


def test_load_local_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("text,label\nhello,1\nworld,0\n")
    cfg = OmegaConf.create({
        "path": str(csv_file),
        "format": "csv",
        "max_samples": None,
    })
    df = _load_local(cfg)
    assert len(df) == 2
    assert "text" in df.columns


def test_load_local_jsonl(tmp_path):
    jsonl_file = tmp_path / "test.jsonl"
    jsonl_file.write_text('{"text": "hello"}\n{"text": "world"}\n')
    cfg = OmegaConf.create({
        "path": str(jsonl_file),
        "format": "jsonl",
        "max_samples": None,
    })
    df = _load_local(cfg)
    assert len(df) == 2
    assert "text" in df.columns


def test_load_local_max_samples(tmp_path):
    csv_file = tmp_path / "big.csv"
    csv_file.write_text("text\n" + "\n".join(f"row{i}" for i in range(100)))
    cfg = OmegaConf.create({
        "path": str(csv_file),
        "format": "csv",
        "max_samples": 10,
    })
    df = _load_local(cfg)
    assert len(df) == 10


def test_load_local_unsupported_format(tmp_path):
    cfg = OmegaConf.create({
        "path": str(tmp_path / "file.xyz"),
        "format": "xyz",
        "max_samples": None,
    })
    with pytest.raises(ValueError, match="Unsupported local file format"):
        _load_local(cfg)
