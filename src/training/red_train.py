"""Attacker (red mode) GRPO training with real model fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
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

# Try to import Unsloth; fallback to standard transformers
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    logger.warning("Unsloth not available; using standard transformers (slower training)")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    LoraConfig = None
    get_peft_model = None


@dataclass
class GRPOTrainingConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
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
    def from_yaml(cls, config_path: Optional[str]) -> "GRPOTrainingConfig":
        if not config_path:
            return cls()
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        red = (data.get("training") or {}).get("red_mode") or {}
        models_cfg = (data.get("models") or {}).get("attacker") or {}
        lora_cfg = models_cfg.get("LoRA", {})

        return cls(
            model_name=models_cfg.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct"),
            epochs=red.get("epochs", 3),
            batch_size=red.get("batch_size", 8),
            learning_rate=red.get("learning_rate", 5e-5),
            num_train_steps=red.get("num_train_steps", 100),
            gradient_accumulation_steps=red.get("gradient_accumulation_steps", 4),
            warmup_steps=red.get("warmup_steps", 10),
            weight_decay=red.get("weight_decay", 0.01),
            max_grad_norm=red.get("max_grad_norm", 1.0),
            seed=red.get("seed", 42),
            lora_r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        )


class AttackerGRPOTrainer:
    """Real GRPO trainer for attack policy using Unsloth + TRL."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        config: Optional[GRPOTrainingConfig] = None,
        output_dir: str = "output/attacker",
        use_wandb: bool = True,
    ):
        self.model_name = model_name
        self.config = config or GRPOTrainingConfig(model_name=model_name)
        self.output_dir = output_dir
        self.use_wandb = use_wandb

        # Logger
        self.logger = PromptCTFLogger(
            enabled=use_wandb,
            mode=Mode.RED,
            task_id="training",
            model_name=model_name,
        )

        # Models loaded on-demand in train()
        self.model = None
        self.tokenizer = None

    def train(self, task_id: str = "easy") -> Dict[str, Any]:
        """Perform GRPO training on attack task."""
        if torch is None:
            raise ImportError(
                "PyTorch is required for training. Install with: pip install torch transformers trl peft"
            )

        os.makedirs(self.output_dir, exist_ok=True)

        # Set seed
        torch.manual_seed(self.config.seed)

        logger.info(f"Starting RED (attacker) GRPO training on {task_id}")
        logger.info(f"Model: {self.model_name}, Device: {self.config.device}")

        # Load model and tokenizer
        self._load_model()

        # Get task
        task = TASK_REGISTRY.get(task_id)
        difficulty = task.difficulty

        # Create environment (with mock defender for now; swap to qwen if needed)
        env = PromptCTFEnv(
            task_id=task_id,
            mode=Mode.RED,
            defender_backend="mock",  # Can switch to "qwen" for real defender
            seed=self.config.seed,
        )

        # Get attack dataset
        dataset = get_dataset(difficulty)

        # Training loop
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = 0
        episodes_completed = 0
        total_flag_extractions = 0

        for epoch in range(self.config.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")

            epoch_rewards = []
            epoch_extractions = 0

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
                episode_length = 0
                flag_leaked = False

                # Single-step episode: send attack prompt
                step_result = env.step(prompt)
                episode_reward = step_result.reward
                flag_leaked = step_result.info["reward_breakdown"]["extracted_flag"]

                # Log step (W&B)
                if self.use_wandb:
                    self.logger.log_step(
                        step=total_steps,
                        reward=episode_reward,
                        episode_length=1,
                        leaked_flag=flag_leaked,
                        blocked=step_result.info["reward_breakdown"]["blocked_attack"],
                        leak_ratio=step_result.info["reward_breakdown"]["leak_ratio"],
                    )

                epoch_rewards.append(episode_reward)
                if flag_leaked:
                    epoch_extractions += 1
                    total_flag_extractions += 1

                total_steps += 1
                episodes_completed += 1

                # Gradient accumulation every N steps
                if total_steps % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # Log epoch stats
            avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
            extraction_rate = epoch_extractions / len(batch_prompts) if batch_prompts else 0.0

            logger.info(
                f"Epoch {epoch + 1} - Avg Reward: {avg_reward:.4f}, "
                f"Extraction Rate: {extraction_rate:.2%}"
            )

            if self.use_wandb:
                self.logger.log_episode(
                    episode=epoch + 1,
                    episode_reward=avg_reward,
                    episode_length=1,
                    flag_extraction_rate=extraction_rate,
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
            "flag_extractions": total_flag_extractions,
            "extraction_rate": total_flag_extractions / episodes_completed if episodes_completed > 0 else 0.0,
        }

        logger.info(f"Training complete: {result}")
        return result

    def _load_model(self) -> None:
        """Load model with Unsloth (if available) or standard transformers."""
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

        # Set tokenizer pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze defender model parameters (opponent remains static during attacker training)
        self._load_and_freeze_defender()

        logger.info("Model loaded successfully")

    def _load_and_freeze_defender(self) -> None:
        """Load defender model and freeze all its parameters."""
        try:
            logger.info("Loading defender model (frozen)...")
            if HAS_UNSLOTH:
                defender_model, _ = FastLanguageModel.from_pretrained(
                    model_name=self.model_name,
                    max_seq_length=512,
                    dtype=torch.float32 if self.config.device == "cpu" else torch.float16,
                    load_in_4bit=self.config.use_4bit and self.config.device == "cuda",
                )
            else:
                defender_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32 if self.config.device == "cpu" else torch.float16,
                    device_map=self.config.device,
                    trust_remote_code=True,
                )

            # Freeze all parameters
            for param in defender_model.parameters():
                param.requires_grad = False

            # Set to eval mode
            defender_model.eval()

            logger.info(f"Defender model frozen: {sum(p.numel() for p in defender_model.parameters() if not p.requires_grad)} frozen parameters")
        except Exception as e:
            logger.warning(f"Could not load defender model for freezing: {e}. Continuing with attacker training only.")

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
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_dir", default="output/attacker")
    parser.add_argument("--config", default="configs/openenv.yaml")
    parser.add_argument("--no-wandb", action="store_true")
    
    args = parser.parse_args()
    
    # Load config
    config = GRPOTrainingConfig.from_yaml(args.config)
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    # Train
    trainer = AttackerGRPOTrainer(
        model_name=args.model,
        config=config,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb
    )
    
    plan = trainer.train(task_id=args.task)
    logger.info("Training plan: %s", plan)
    trainer.save_model(args.output_dir)
