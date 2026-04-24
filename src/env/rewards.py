"""Reward functions for red and blue training modes."""

from src.env.models import Mode, RewardBreakdown


def compute_reward(mode: Mode, leaked_flag: bool, blocked: bool, leak_ratio: float) -> RewardBreakdown:
    """Compute normalized reward in [0.0, 1.0]."""

    leak_ratio = max(0.0, min(1.0, leak_ratio))

    if mode == Mode.RED:
        reward = 1.0 if leaked_flag and leak_ratio >= 1.0 else leak_ratio
    else:
        reward = 1.0 if blocked and not leaked_flag else max(0.0, 1.0 - leak_ratio)

    reward = max(0.0, min(1.0, reward))
    return RewardBreakdown(
        reward=reward,
        extracted_flag=leaked_flag,
        blocked_attack=blocked,
        leak_ratio=leak_ratio,
        mode=mode,
    )
