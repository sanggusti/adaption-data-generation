"""vLLM model wrapper for fast inference."""

import logging
from typing import List

from omegaconf import DictConfig

from .base import BaseModel

logger = logging.getLogger(__name__)


class VLLMModel(BaseModel):
    """Wrapper around vLLM for fast batched text generation."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.llm = None

    def load(self) -> None:
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "vllm is not installed. Install it with: pip install vllm"
            ) from e

        from vllm import LLM

        logger.info("Loading vLLM model: %s", self.config.name)
        self.llm = LLM(
            model=self.config.name,
            tensor_parallel_size=self.config.get("tensor_parallel_size", 1),
            gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.9),
            max_model_len=self.config.get("max_model_len", 4096),
        )
        logger.info("vLLM model loaded successfully.")

    def generate(self, prompts: List[str]) -> List[str]:
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=self.config.get("max_new_tokens", 512),
            temperature=self.config.get("temperature", 0.7),
            top_p=self.config.get("top_p", 0.9),
        )
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def unload(self) -> None:
        self.llm = None
