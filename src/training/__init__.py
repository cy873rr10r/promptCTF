"""Training scaffolds for PromptCTF."""

from src.training.blue_train import DefenderTrainer
from src.training.red_train import AttackerGRPOTrainer

__all__ = ["AttackerGRPOTrainer", "DefenderTrainer"]
