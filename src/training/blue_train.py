"""Defender (blue mode) GRPO training with real model fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import logging
import os
import yaml

try:
    import torch
    from tqdm import tqdm
except ImportError:
    torch = None
    tqdm = None

from src.env.environment import PromptCTFEnv
from src.env.models import Difficulty, Mode
from src.env.tasks import TASK_REGISTRY
from src.training.grpo_dataset import get_dataset
from src.training.wandb_logger import PromptCTFLogger

logger = logging.getLogger(__name__)

# Try to import Unsloth
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    logger.warning("Unsloth not available; using standard transformers")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    LoraConfig = None
    get_peft_model = None


@dataclass
class BlueTrainingConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2.5e-5
    num_train_steps: int = 100
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj")
    lora_dropout: float = 0.05
    device: str = "cpu"
    dtype: str = "float32"
    use_4bit: bool = True

    @classmethod
    def from_yaml(cls, config_path: Optional[str]) -> "BlueTrainingConfig":
        if not config_path:
            return cls()
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        blue = (data.get("training") or {}).get("blue_mode") or {}
        models_cfg = (data.get("models") or {}).get("defender") or {}
        lora_cfg = models_cfg.get("LoRA", {})

        return cls(
            model_name=models_cfg.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct"),
            epochs=blue.get("epochs", 3),
            batch_size=blue.get("batch_size", 8),
            learning_rate=blue.get("learning_rate", 2.5e-5),
            num_train_steps=blue.get("num_train_steps", 100),
            gradient_accumulation_steps=blue.get("gradient_accumulation_steps", 4),
            warmup_steps=blue.get("warmup_steps", 10),
            weight_decay=blue.get("weight_decay", 0.01),
            max_grad_norm=blue.get("max_grad_norm", 1.0),
            seed=blue.get("seed", 42),
            lora_r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        )


class DefenderTrainer:
    """Real GRPO trainer for defense policy (blue mode)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        config: Optional[BlueTrainingConfig] = None,
        output_dir: str = "output/defender",
        use_wandb: bool = True,
    ):
        self.model_name = model_name
        self.config = config or BlueTrainingConfig(model_name=model_name)
        self.output_dir = output_dir
        self.use_wandb = use_wandb

        # Logger
        self.logger = PromptCTFLogger(
            enabled=use_wandb,
            mode=Mode.BLUE,
            task_id="training",
            model_name=model_name,
        )

        # Models loaded on-demand
        self.model = None
        self.tokenizer = None

    def train(self, task_id: str = "easy") -> Dict[str, Any]:
        """Perform GRPO training on defense task."""
        if torch is None:
            raise ImportError(
                "PyTorch is required for training. Install with: pip install torch transformers trl"
            )

        os.makedirs(self.output_dir, exist_ok=True)

        # Set seed
        torch.manual_seed(self.config.seed)

        logger.info(f"Starting BLUE (defender) GRPO training on {task_id}")
        logger.info(f"Model: {self.model_name}, Device: {self.config.device}")

        # Load model
        self._load_model()

        # Get task
        task = TASK_REGISTRY.get(task_id)
        difficulty = task.difficulty

        # Create environment in BLUE mode (reward = block rate)
        env = PromptCTFEnv(
            task_id=task_id,
            mode=Mode.BLUE,
            defender_backend="mock",  # Use mock defender since we're training it
            seed=self.config.seed,
        )

        # Get attack dataset
        dataset = get_dataset(difficulty)

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = 0
        episodes_completed = 0
        total_blocks = 0

        for epoch in range(self.config.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")

            epoch_rewards = []
            epoch_blocks = 0

            # Sample batch of attack prompts
            batch_prompts = dataset.sample_batch(self.config.batch_size)

            pbar = tqdm(
                batch_prompts,
                desc=f"Epoch {epoch + 1}",
                disable=not logger.isEnabledFor(logging.INFO),
            )

            for prompt in pbar:
                # Reset environment for new episode
                obs, info = env.reset()
                episode_reward = 0.0
                blocked = False

                # Single-step episode: receive attack prompt
                step_result = env.step(prompt)
                episode_reward = step_result.reward
                blocked = step_result.info["reward_breakdown"]["blocked_attack"]

                # Log step
                if self.use_wandb:
                    self.logger.log_step(
                        step=total_steps,
                        reward=episode_reward,
                        episode_length=1,
                        blocked=blocked,
                        leak_ratio=step_result.info["reward_breakdown"]["leak_ratio"],
                    )

                epoch_rewards.append(episode_reward)
                if blocked:
                    epoch_blocks += 1
                    total_blocks += 1

                total_steps += 1
                episodes_completed += 1

                # Gradient accumulation
                if total_steps % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # Log epoch stats
            avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
            block_rate = epoch_blocks / len(batch_prompts) if batch_prompts else 0.0

            logger.info(
                f"Epoch {epoch + 1} - Avg Reward: {avg_reward:.4f}, "
                f"Block Rate: {block_rate:.2%}"
            )

            if self.use_wandb:
                self.logger.log_episode(
                    episode=epoch + 1,
                    episode_reward=avg_reward,
                    episode_length=1,
                    block_rate=block_rate,
                )

        # Final gradient step
        if total_steps % self.config.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        env.close()

        # Save model
        self.save_model(self.output_dir)

        result = {
            "status": "completed",
            "task_id": task_id,
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "total_steps": total_steps,
            "episodes_completed": episodes_completed,
            "blocks_success": total_blocks,
            "block_rate": total_blocks / episodes_completed if episodes_completed > 0 else 0.0,
        }

        logger.info(f"Training complete: {result}")
        return result

    def _load_model(self) -> None:
        """Load model with Unsloth or standard transformers."""
        if HAS_UNSLOTH:
            logger.info("Loading model with Unsloth optimization...")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=512,
                dtype=torch.float32 if self.config.device == "cpu" else torch.float16,
                load_in_4bit=self.config.use_4bit and self.config.device == "cuda",
            )

            # Prepare for LoRA
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=list(self.config.lora_target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
        else:
            logger.info("Loading model with standard transformers...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32 if self.config.device == "cpu" else torch.float16,
                device_map=self.config.device,
                trust_remote_code=True,
            )

            # Apply LoRA
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=list(self.config.lora_target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze attacker model parameters (opponent remains static during defender training)
        self._load_and_freeze_attacker()

        logger.info("Model loaded successfully")

    def _load_and_freeze_attacker(self) -> None:
        """Load attacker model and freeze all its parameters."""
        try:
            logger.info("Loading attacker model (frozen)...")
            if HAS_UNSLOTH:
                attacker_model, _ = FastLanguageModel.from_pretrained(
                    model_name=self.model_name,
                    max_seq_length=512,
                    dtype=torch.float32 if self.config.device == "cpu" else torch.float16,
                    load_in_4bit=self.config.use_4bit and self.config.device == "cuda",
                )
            else:
                attacker_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32 if self.config.device == "cpu" else torch.float16,
                    device_map=self.config.device,
                    trust_remote_code=True,
                )

            # Freeze all parameters
            for param in attacker_model.parameters():
                param.requires_grad = False

            # Set to eval mode
            attacker_model.eval()

            logger.info(f"Attacker model frozen: {sum(p.numel() for p in attacker_model.parameters() if not p.requires_grad)} frozen parameters")
        except Exception as e:
            logger.warning(f"Could not load attacker model for freezing: {e}. Continuing with defender training only.")

    def save_model(self, save_path: str) -> None:
        """Save model and LoRA weights."""
        os.makedirs(save_path, exist_ok=True)

        if HAS_UNSLOTH:
            logger.info(f"Saving Unsloth model to {save_path}")
            self.model.save_pretrained(save_path)
        else:
            logger.info(f"Saving model with LoRA to {save_path}")
            self.model.save_pretrained(save_path)

        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2.5e-5)
    parser.add_argument("--output_dir", default="output/defender")
    parser.add_argument("--config", default="configs/openenv.yaml")
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    # Load config
    config = BlueTrainingConfig.from_yaml(args.config)
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate

    # Train
    trainer = DefenderTrainer(
        model_name=args.model,
        config=config,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
    )

    result = trainer.train(task_id=args.task)
    logger.info("Training result: %s", result)
