"""Run training scripts"""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.red_train import AttackerGRPOTrainer, GRPOTrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train attacker model with GRPO")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy",
                       help="Task to train on")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--output_dir", default="output/attacker")
    parser.add_argument("--config", default="configs/openenv.yaml")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = GRPOTrainingConfig(args.config)
    
    # Override with command line args
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    # Create trainer
    logger.info(f"Creating trainer for task: {args.task}")
    trainer = AttackerGRPOTrainer(
        model_name=args.model,
        config=config,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train(task_id=args.task)
    
    # Save
    trainer.save_model(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
