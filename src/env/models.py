"""Typed schemas for PromptCTF environment interactions."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal


Role = Literal["system", "user", "assistant"]


class Difficulty(str, Enum):
    """Task difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Mode(str, Enum):
    """Training/evaluation mode."""

    RED = "red"
    BLUE = "blue"


@dataclass
class Message:
    """Conversation message."""

    role: Role
    content: str


@dataclass
class TaskSpec:
    """Prompt injection CTF task specification."""

    task_id: str
    name: str
    description: str
    difficulty: Difficulty
    system_prompt: str
    flag: str
    max_steps: int


@dataclass
class Action:
    """Attacker action payload."""

    prompt: str


@dataclass
class Observation:
    """State visible to the acting policy/UI."""

    task_id: str
    task_description: str
    mode: Mode
    step: int
    max_steps: int
    conversation: List[Message]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["conversation"] = [asdict(msg) for msg in self.conversation]
        payload["mode"] = self.mode.value
        return payload


@dataclass
class DefenderTurn:
    """Defender response + decision metadata."""

    response: str
    blocked: bool
    leaked_flag: bool
    leak_ratio: float


@dataclass
class RewardBreakdown:
    """Normalized reward components in [0.0, 1.0]."""

    reward: float
    extracted_flag: bool
    blocked_attack: bool
    leak_ratio: float
    mode: Mode

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload


@dataclass
class StepResult:
    """Single environment transition."""

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info,
        }
