"""W&B integration for PromptCTF-Env training."""

import wandb
import logging
from typing import Optional, Dict, Any
from src.env.models import Difficulty, Mode

logger = logging.getLogger(__name__)


class PromptCTFLogger:
    """Unified W&B logger for red/blue training modes."""

    def __init__(
        self,
        project: str = "promptctf-env",
        entity: Optional[str] = None,
        enabled: bool = True,
        mode: Mode = Mode.RED,
        task_id: str = "easy",
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    ):
        """Initialize W&B logger.

        Args:
            project: W&B project name
            entity: W&B entity (team/username)
            enabled: Whether to enable W&B logging
            mode: RED (attacker) or BLUE (defender)
            task_id: Task ID (easy/medium/hard)
            model_name: Model being trained
        """
        self.enabled = enabled
        self.mode = mode
        self.task_id = task_id
        self.model_name = model_name

        if self.enabled:
            run_name = f"{mode.value}-{task_id}-{model_name.split('/')[-1]}"
            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config={
                    "mode": mode.value,
                    "task_id": task_id,
                    "model_name": model_name,
                },
            )
            logger.info(f"W&B initialized: {project}/{run_name}")

    def log_step(
        self,
        step: int,
        reward: float,
        episode_length: int,
        leaked_flag: bool = False,
        blocked: bool = False,
        leak_ratio: float = 0.0,
        **kwargs,
    ) -> None:
        """Log per-step metrics.

        Args:
            step: Global step number
            reward: Step reward [0.0, 1.0]
            episode_length: Current episode length
            leaked_flag: Whether flag was leaked (RED mode)
            blocked: Whether attack was blocked (BLUE mode)
            leak_ratio: Fraction of flag leaked [0.0, 1.0]
            **kwargs: Additional metrics
        """
        if not self.enabled:
            return

        metrics = {
            "step": step,
            "reward": reward,
            "episode_length": episode_length,
            "task_difficulty": self.task_id,
            "mode": self.mode.value,
            "leaked_flag": leaked_flag,
            "blocked": blocked,
            "leak_ratio": leak_ratio,
        }
        metrics.update(kwargs)
        wandb.log(metrics, step=step)

    def log_episode(
        self,
        episode: int,
        episode_reward: float,
        episode_length: int,
        flag_extraction_rate: float = 0.0,
        block_rate: float = 0.0,
        avg_leak_ratio: float = 0.0,
        **kwargs,
    ) -> None:
        """Log per-episode metrics.

        Args:
            episode: Episode number
            episode_reward: Total reward for episode
            episode_length: Steps in episode
            flag_extraction_rate: Success rate (RED mode)
            block_rate: Block success rate (BLUE mode)
            avg_leak_ratio: Average leak ratio across steps
            **kwargs: Additional metrics
        """
        if not self.enabled:
            return

        metrics = {
            "episode": episode,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            f"{self.mode.value}/flag_extraction_rate": flag_extraction_rate,
            f"{self.mode.value}/block_rate": block_rate,
            f"{self.mode.value}/avg_leak_ratio": avg_leak_ratio,
        }
        metrics.update(kwargs)
        wandb.log(metrics, step=episode)

    def log_batch_stats(
        self,
        batch_num: int,
        avg_reward: float,
        success_rate: float,
        avg_leak_ratio: float,
        batch_size: int,
        **kwargs,
    ) -> None:
        """Log batch-level statistics.

        Args:
            batch_num: Batch number
            avg_reward: Average reward in batch
            success_rate: Success rate in batch
            avg_leak_ratio: Average leak ratio
            batch_size: Size of batch
            **kwargs: Additional metrics
        """
        if not self.enabled:
            return

        metrics = {
            "batch": batch_num,
            "batch_avg_reward": avg_reward,
            "batch_success_rate": success_rate,
            "batch_avg_leak_ratio": avg_leak_ratio,
            "batch_size": batch_size,
        }
        metrics.update(kwargs)
        wandb.log(metrics, step=batch_num)

    def log_model_checkpoint(self, save_path: str, step: int) -> None:
        """Log model checkpoint artifact.

        Args:
            save_path: Path to saved model
            step: Training step
        """
        if not self.enabled:
            return

        artifact = wandb.Artifact(
            f"model-{self.mode.value}-{self.task_id}-step{step}",
            type="model",
        )
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)
        logger.info(f"Logged checkpoint: {save_path}")

    def log_hyperparams(self, config: Dict[str, Any]) -> None:
        """Log training hyperparameters.

        Args:
            config: Configuration dictionary
        """
        if not self.enabled:
            return

        wandb.config.update(config)
        logger.info("Updated W&B config with hyperparameters")

    def finish(self) -> None:
        """Finish W&B run."""
        if self.enabled:
            wandb.finish()
            logger.info("W&B run finished")
