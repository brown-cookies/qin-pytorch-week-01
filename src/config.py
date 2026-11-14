from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Config:
    seed: int = 42
    test_size_split: float = 0.15
    val_size_split: float = 0.15
    epochs: int = 2000
    learning_rate: float = 0.01


config = Config()
