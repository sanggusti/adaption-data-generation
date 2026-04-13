"""HuggingFace Transformers model wrapper."""

import logging
from typing import List

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .base import BaseModel

logger = logging.getLogger(__name__)


class HFTransformersModel(BaseModel):
    """Wrapper around HuggingFace Transformers for text generation."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.pipe = None

    def load(self) -> None:
        logger.info("Loading HuggingFace model: %s", self.config.name)
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.get("torch_dtype", "float16"), torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.name,
            torch_dtype=torch_dtype,
            device_map=self.config.get("device", "auto"),
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        logger.info("Model loaded successfully.")

    def generate(self, prompts: List[str]) -> List[str]:
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        results = self.pipe(
            prompts,
            max_new_tokens=self.config.get("max_new_tokens", 512),
            temperature=self.config.get("temperature", 0.7),
            top_p=self.config.get("top_p", 0.9),
            do_sample=self.config.get("do_sample", True),
            batch_size=self.config.get("batch_size", 4),
        )
        return [r[0]["generated_text"] for r in results]

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
