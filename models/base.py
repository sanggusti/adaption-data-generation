"""Base model interface for all inference backends."""

from abc import ABC, abstractmethod
from typing import List, Union


class BaseModel(ABC):
    """Abstract base class for all model inference backends."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        ...

    @abstractmethod
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for a list of prompts."""
        ...

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.unload()

    def unload(self) -> None:
        """Unload the model from memory (optional)."""
        pass
