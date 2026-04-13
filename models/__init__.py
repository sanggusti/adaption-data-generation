"""Model inference backends."""

from .base import BaseModel
from .hf_model import HFTransformersModel
from .unsloth_model import UnslothModel
from .vllm_model import VLLMModel

_MODEL_REGISTRY = {
    "hf_transformers": HFTransformersModel,
    "vllm": VLLMModel,
    "unsloth": UnslothModel,
}


def get_model(config) -> BaseModel:
    """Instantiate the model backend specified in config."""
    model_type = config.model.type
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Choose from: {list(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[model_type](config.model)


__all__ = [
    "BaseModel",
    "HFTransformersModel",
    "VLLMModel",
    "UnslothModel",
    "get_model",
]
