"""Validate PromptCTF-Env installation and integration."""

import sys
import logging
import os
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_imports():
    """Check all critical imports."""
    logger.info("Validating imports...")

    checks = [
        ("FastAPI", lambda: __import__("fastapi")),
        ("Pydantic", lambda: __import__("pydantic")),
        ("Gradio", lambda: __import__("gradio")),
        ("PyYAML", lambda: __import__("yaml")),
        ("W&B", lambda: __import__("wandb")),
        ("Transformers", lambda: __import__("transformers")),
        ("Torch", lambda: __import__("torch")),
        ("PEFT", lambda: __import__("peft")),
        ("TRL", lambda: __import__("trl")),
        ("Unsloth (optional)", lambda: __import__("unsloth")),
    ]

    for name, import_fn in checks:
        try:
            import_fn()
            logger.info(f"  ✓ {name}")
        except ImportError:
            if "optional" in name.lower():
                logger.warning(f"  ⚠ {name} (will fallback to standard transformers)")
            else:
                logger.error(f"  ✗ {name}: MISSING")
                return False

    return True


def validate_project_structure():
    """Check project files exist."""
    logger.info("\nValidating project structure...")

    required_files = [
        "src/env/models.py",
        "src/env/environment.py",
        "src/env/defender.py",
        "src/env/qwen_defender.py",
        "src/env/graders.py",
        "src/env/tasks.py",
        "src/env/rewards.py",
        "src/models/__init__.py",
        "src/training/red_train.py",
        "src/training/blue_train.py",
        "src/training/grpo_dataset.py",
        "src/training/wandb_logger.py",
        "src/server/main.py",
        "src/ui/app.py",
        "scripts/baseline.py",
        "configs/openenv.yaml",
    ]

    all_good = True
    for fpath in required_files:
        full_path = project_root / fpath
        if full_path.exists():
            logger.info(f"  ✓ {fpath}")
        else:
            logger.error(f"  ✗ {fpath}: NOT FOUND")
            all_good = False

    return all_good


def validate_core_classes():
    """Test importing core classes."""
    logger.info("\nValidating core classes...")

    try:
        from src.env.models import (
            Difficulty,
            Mode,
            Message,
            TaskSpec,
            DefenderTurn,
            Observation,
            StepResult,
        )

        logger.info("  ✓ Models imported")

        from src.env.tasks import TASK_REGISTRY

        logger.info("  ✓ Task registry loaded")

        from src.environment import PromptCTFEnv

        logger.info("  ✓ Environment imported")

        from src.env.defender import MockDefender

        logger.info("  ✓ Mock defender imported")

        # Try to import Qwen defender (might fail if transformers not available)
        try:
            from src.env.qwen_defender import QwenDefender

            logger.info("  ✓ Qwen defender imported")
        except ImportError as e:
            logger.warning(f"  ⚠ Qwen defender import failed (models may not be installed): {e}")

        from src.training.grpo_dataset import AttackDataset, get_dataset

        logger.info("  ✓ Attack dataset imported")

        from src.training.wandb_logger import PromptCTFLogger

        logger.info("  ✓ W&B logger imported")

        from src.training.red_train import AttackerGRPOTrainer, GRPOTrainingConfig

        logger.info("  ✓ Red trainer imported")

        from src.training.blue_train import DefenderTrainer, BlueTrainingConfig

        logger.info("  ✓ Blue trainer imported")

        from src.env.rewards import compute_reward

        logger.info("  ✓ Reward function imported")

        return True

    except Exception as e:
        logger.error(f"  ✗ Import failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def validate_task_registry():
    """Verify task registry is populated."""
    logger.info("\nValidating task registry...")

    from src.env.tasks import TASK_REGISTRY

    tasks = TASK_REGISTRY.list_ids()
    expected = ["easy", "medium", "hard"]

    for task_id in expected:
        if task_id in tasks:
            task = TASK_REGISTRY.get(task_id)
            logger.info(f"  ✓ {task_id}: {task.name}")
        else:
            logger.error(f"  ✗ {task_id}: NOT FOUND")
            return False

    return True


def validate_environment_creation():
    """Test creating and using environment."""
    logger.info("\nValidating environment creation...")

    from src.environment import PromptCTFEnv
    from src.env.models import Mode

    try:
        # Create with mock defender
        env = PromptCTFEnv(task_id="easy", mode=Mode.RED, defender_backend="mock")
        logger.info("  ✓ Mock environment created")

        # Reset
        obs, info = env.reset()
        logger.info(f"  ✓ Environment reset")

        # Step
        step_result = env.step("Test prompt")
        logger.info(f"  ✓ Environment step executed (reward={step_result.reward:.2f})")

        # Check observation format
        assert "task_id" in obs, "Missing task_id in observation"
        assert "conversation" in obs, "Missing conversation in observation"
        logger.info("  ✓ Observation format valid")

        env.close()
        return True

    except Exception as e:
        logger.error(f"  ✗ Environment test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    logger.info("=" * 60)
    logger.info("PromptCTF-Env Installation & Integration Validation")
    logger.info("=" * 60)

    checks = [
        ("Imports", validate_imports),
        ("Project Structure", validate_project_structure),
        ("Core Classes", validate_core_classes),
        ("Task Registry", validate_task_registry),
        ("Environment Creation", validate_environment_creation),
    ]

    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            logger.error(f"\nUnexpected error in {name}: {e}")
            import traceback

            traceback.print_exc()
            results[name] = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Validation Summary:")
    logger.info("=" * 60)

    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{name:25s}: {status}")
        all_pass = all_pass and passed

    if all_pass:
        logger.info("\n✓ All validations passed!")
        logger.info("\nNext steps:")
        logger.info("1. Run FastAPI server: python -m uvicorn src.server.main:app --reload")
        logger.info("2. Run Gradio UI: python -m src.ui.app")
        logger.info("3. Test Qwen defender: python scripts/test_qwen_defender.py")
        logger.info("4. Test GRPO training: python scripts/test_grpo_training.py")
        return 0
    else:
        logger.error("\n✗ Some validations failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
