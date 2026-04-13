"""Main entry point for adaption-data-generation.

Run with:
    python main.py dataset=huggingface model=hf_transformers generation=default
    python main.py dataset=kaggle model=vllm generation=adaption
"""

import logging

import hydra
from omegaconf import DictConfig

from src.generation.generator import DataGenerator
from src.utils.helpers import set_seed

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.get("seed", 42))
    log.info("Starting data generation experiment: %s", cfg.get("experiment_name"))
    generator = DataGenerator(cfg)
    df = generator.run()
    log.info("Generation complete. Produced %d samples.", len(df))


if __name__ == "__main__":
    main()
