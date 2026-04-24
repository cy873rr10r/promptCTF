"""Baseline evaluation script: runs all tasks with real Qwen2.5-0.5B defender via Unsloth."""

import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment import PromptCTFEnv
from src.env.models import Mode
from src.env.tasks import TASK_REGISTRY
from src.training.grpo_dataset import get_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_baseline() -> Dict[str, Dict]:
    """Run baseline with real Qwen2.5-0.5B-Instruct defender."""
    results = {}

    for task_id in ["easy", "medium", "hard"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Task: {task_id}")
        logger.info(f"{'='*60}")

        task = TASK_REGISTRY.get(task_id)

        # Create environment with real Qwen defender
        env = PromptCTFEnv(
            task_id=task_id,
            mode=Mode.RED,
            defender_backend="qwen",
            seed=42,
        )

        obs, info = env.reset()

        # Try a simple attack
        attack_dataset = get_dataset(task.difficulty)
        attack_prompt = attack_dataset.sample_prompt()

        logger.info(f"Attack prompt: {attack_prompt[:60]}...")

        step_result = env.step(attack_prompt)

        reward = step_result.reward
        leaked_flag = step_result.info["reward_breakdown"]["extracted_flag"]
        leak_ratio = step_result.info["reward_breakdown"]["leak_ratio"]
        steps_taken = 1

        logger.info(f"Reward: {reward:.4f}")
        logger.info(f"Flag leaked: {leaked_flag}")
        logger.info(f"Leak ratio: {leak_ratio:.2f}")
        logger.info(f"Steps taken: {steps_taken}")

        results[task_id] = {
            "task_id": task_id,
            "difficulty": task.difficulty.value,
            "reward": round(reward, 4),
            "leaked_flag": leaked_flag,
            "leak_ratio": round(leak_ratio, 4),
            "steps_taken": steps_taken,
        }

        env.close()

    return results


def print_results(results: Dict[str, Dict]) -> None:
    """Print formatted baseline results."""
    logger.info(f"\n{'='*60}")
    logger.info("BASELINE RESULTS")
    logger.info(f"{'='*60}")

    header = f"{'Task':<12} {'Difficulty':<12} {'Reward':<10} {'Steps':<8}"
    logger.info(header)
    logger.info("-" * 60)

    total_reward = 0.0
    for task_id, result in results.items():
        reward = result["reward"]
        difficulty = result["difficulty"]
        steps = result["steps_taken"]
        total_reward += reward

        line = (
            f"{task_id:<12} {difficulty:<12} {reward:<10.4f} {steps:<8}"
        )
        logger.info(line)

    logger.info("-" * 60)
    avg_reward = total_reward / len(results)
    logger.info(f"{'Average':<12} {'':<12} {avg_reward:<10.4f}")
    logger.info(f"{'='*60}")


def save_results(results: Dict[str, Dict], output_file: str = "baseline_results.json") -> None:
    """Save baseline results to JSON."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")


def main() -> int:
    """Run baseline and report results."""
    logger.info("Starting PromptCTF-Env Baseline Evaluation")
    logger.info("Model: Qwen2.5-0.5B-Instruct (real inference)")
    logger.info("Mode: RED (attacker)")

    try:
        results = run_baseline()
        print_results(results)
        save_results(results)
        logger.info("\n✓ Baseline evaluation complete")
        return 0
    except Exception as e:
        logger.error(f"\n✗ Baseline evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
