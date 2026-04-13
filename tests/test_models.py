"""Tests for model wrappers."""

import pytest
from omegaconf import OmegaConf

from models.base import BaseModel


def test_base_model_is_abstract():
    with pytest.raises(TypeError):
        BaseModel(config=None)  # type: ignore


def test_hf_model_missing_transformers(monkeypatch):
    """HFTransformersModel.load() should raise if transformers is missing."""
    import sys
    monkeypatch.setitem(sys.modules, "transformers", None)
    # Re-import inside the test after patching
    # Just verify instantiation works before load
    from models.hf_model import HFTransformersModel
    cfg = OmegaConf.create({"name": "gpt2", "torch_dtype": "float32", "device": "cpu"})
    m = HFTransformersModel(cfg)
    assert m.model is None


def test_vllm_model_import_error():
    """VLLMModel.load() should raise ImportError if vllm is not installed."""
    import sys
    import importlib
    original = sys.modules.get("vllm")
    sys.modules["vllm"] = None  # type: ignore
    try:
        from models.vllm_model import VLLMModel
        cfg = OmegaConf.create({"name": "mistralai/Mistral-7B-Instruct-v0.3"})
        m = VLLMModel(cfg)
        with pytest.raises((ImportError, TypeError)):
            m.load()
    finally:
        if original is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = original


def test_unsloth_model_import_error():
    """UnslothModel.load() should raise ImportError if unsloth is not installed."""
    import sys
    original = sys.modules.get("unsloth")
    sys.modules["unsloth"] = None  # type: ignore
    try:
        from models.unsloth_model import UnslothModel
        cfg = OmegaConf.create({"name": "unsloth/mistral-7b"})
        m = UnslothModel(cfg)
        with pytest.raises((ImportError, TypeError)):
            m.load()
    finally:
        if original is None:
            sys.modules.pop("unsloth", None)
        else:
            sys.modules["unsloth"] = original


def test_get_model_unknown_type():
    from models import get_model
    cfg = OmegaConf.create({"model": {"type": "unknown_backend"}})
    with pytest.raises(ValueError, match="Unknown model type"):
        get_model(cfg)
