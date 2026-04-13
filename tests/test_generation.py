"""Tests for the DataGenerator pipeline."""

import json
import os

import pandas as pd
import pytest
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch

from src.generation.generator import DataGenerator
from src.data.processors import build_prompts, deduplicate


def _make_config(tmp_path, csv_path):
    return OmegaConf.create({
        "dataset": {
            "type": "local",
            "path": str(csv_path),
            "format": "csv",
            "max_samples": None,
            "text_column": "text",
        },
        "model": {
            "type": "hf_transformers",
            "name": "gpt2",
        },
        "generation": {
            "type": "default",
            "prompt_template": "Answer: {text}",
            "num_samples": None,
            "output_format": "jsonl",
            "deduplicate": True,
        },
        "output_dir": str(tmp_path / "outputs"),
        "experiment_name": "test_exp",
        "seed": 42,
    })


def test_build_prompts():
    df = pd.DataFrame({"text": ["hello", "world"]})
    cfg = OmegaConf.create({
        "dataset": {"text_column": "text"},
        "generation": {"prompt_template": "Say: {text}"},
    })
    prompts = build_prompts(df, cfg)
    assert prompts == ["Say: hello", "Say: world"]


def test_build_prompts_missing_column():
    df = pd.DataFrame({"other": ["a", "b"]})
    cfg = OmegaConf.create({
        "dataset": {"text_column": "text"},
        "generation": {"prompt_template": "{text}"},
    })
    with pytest.raises(ValueError, match="text_column"):
        build_prompts(df, cfg)


def test_deduplicate():
    df = pd.DataFrame({"text": ["a", "a", "b"]})
    out = deduplicate(df, column="text")
    assert len(out) == 2


def test_data_generator_run(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("text\nhello\nworld\nhello\n")
    cfg = _make_config(tmp_path, csv_file)

    fake_model = MagicMock()
    fake_model.__enter__ = lambda s: s
    fake_model.__exit__ = MagicMock(return_value=False)
    fake_model.generate.return_value = ["response1", "response2"]

    with patch("src.generation.generator.get_model", return_value=fake_model):
        gen = DataGenerator(cfg)
        df = gen.run()

    assert "generated_response" in df.columns
    assert "prompt" in df.columns
    out_file = os.path.join(str(tmp_path / "outputs"), "test_exp.jsonl")
    assert os.path.exists(out_file)


def test_data_generator_save_csv(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("text\nhello\nworld\n")
    cfg = _make_config(tmp_path, csv_file)
    cfg.generation.output_format = "csv"

    fake_model = MagicMock()
    fake_model.__enter__ = lambda s: s
    fake_model.__exit__ = MagicMock(return_value=False)
    fake_model.generate.return_value = ["r1", "r2"]

    with patch("src.generation.generator.get_model", return_value=fake_model):
        gen = DataGenerator(cfg)
        df = gen.run()

    out_file = os.path.join(str(tmp_path / "outputs"), "test_exp.csv")
    assert os.path.exists(out_file)
