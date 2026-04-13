"""Unsloth model wrapper for efficient fine-tuned inference."""

import logging
from typing import List

from omegaconf import DictConfig

from .base import BaseModel

logger = logging.getLogger(__name__)


class UnslothModel(BaseModel):
    """Wrapper around Unsloth for efficient quantized inference."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        try:
            from unsloth import FastLanguageModel  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "unsloth is not installed. Install it with: pip install unsloth"
            ) from e

        from unsloth import FastLanguageModel

        logger.info("Loading Unsloth model: %s", self.config.name)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.name,
            max_seq_length=self.config.get("max_seq_length", 2048),
            dtype=self.config.get("dtype", None),
            load_in_4bit=self.config.get("load_in_4bit", True),
        )
        FastLanguageModel.for_inference(self.model)
        logger.info("Unsloth model loaded successfully.")

    def generate(self, prompts: List[str]) -> List[str]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        import torch

        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get("max_new_tokens", 512),
                    temperature=self.config.get("temperature", 0.7),
                    top_p=self.config.get("top_p", 0.9),
                    do_sample=True,
                )
            results.append(
                self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            )
        return results

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
