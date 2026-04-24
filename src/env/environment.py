"""Lightweight PromptCTF environment for local logic testing."""

from typing import Any, Dict, List, Optional, Tuple

from src.env.defender import MockDefender
from src.env.qwen_defender import QwenDefender
from src.env.models import Action, Message, Mode, Observation, StepResult, TaskSpec
from src.env.rewards import compute_reward
from src.env.tasks import TASK_REGISTRY


class PromptCTFEnv:
    """OpenEnv-style interface: reset() and step()."""

    def __init__(
        self,
        task_id: str = "easy",
        mode: Mode = Mode.RED,
        defender_backend: str = "mock",
        seed: Optional[int] = None,
    ) -> None:
        if defender_backend not in ("mock", "qwen"):
            raise ValueError(
                f"Unsupported defender backend: {defender_backend}. "
                "Supported: 'mock', 'qwen'"
            )

        self.task: TaskSpec = TASK_REGISTRY.get(task_id)
        self.mode = mode
        self.defender_backend = defender_backend
        self.max_steps = self.task.max_steps

        # Load appropriate defender
        if defender_backend == "qwen":
            import torch; self.defender = QwenDefender(device="cuda" if torch.cuda.is_available() else "cpu")
        else:  # mock
            self.defender = MockDefender(seed=seed)

        self.current_step = 0
        self.done = False
        self.episode_reward = 0.0
        self.conversation: List[Message] = []

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.defender = MockDefender(seed=seed)

        self.current_step = 0
        self.done = False
        self.episode_reward = 0.0
        self.conversation = [Message(role="system", content=self.task.system_prompt)]

        observation = self._build_observation().to_dict()
        info = {
            "task_id": self.task.task_id,
            "difficulty": self.task.difficulty.value,
            "mode": self.mode.value,
            "max_steps": self.max_steps,
        }
        return observation, info

    def step(self, action: str | Action) -> StepResult:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        if isinstance(action, Action):
            prompt = action.prompt
        else:
            prompt = action

        self.current_step += 1
        self.conversation.append(Message(role="user", content=prompt))

        defender_turn = self.defender.respond(self.task, self.conversation, prompt)
        self.conversation.append(Message(role="assistant", content=defender_turn.response))

        reward_breakdown = compute_reward(
            mode=self.mode,
            leaked_flag=defender_turn.leaked_flag,
            blocked=defender_turn.blocked,
            leak_ratio=defender_turn.leak_ratio,
        )

        reward = reward_breakdown.reward
        self.episode_reward += reward

        terminated = defender_turn.leaked_flag and self.mode == Mode.RED
        truncated = self.current_step >= self.max_steps
        self.done = terminated or truncated

        info: Dict[str, Any] = {
            "defender_response": defender_turn.response,
            "reward_breakdown": reward_breakdown.to_dict(),
            "episode_reward": round(self.episode_reward, 4),
            "step": self.current_step,
            "max_steps": self.max_steps,
        }

        return StepResult(
            observation=self._build_observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _build_observation(self) -> Observation:
        return Observation(
            task_id=self.task.task_id,
            task_description=self.task.description,
            mode=self.mode,
            step=self.current_step,
            max_steps=self.max_steps,
            conversation=self.conversation.copy(),
        )

    def close(self) -> None:
        self.done = True
