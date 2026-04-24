"""Typed models for PromptCTF."""
from enum import Enum

class Mode(str, Enum):
    RED = "red"
    BLUE = "blue"

class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
