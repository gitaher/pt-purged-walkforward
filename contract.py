"""Structured tool and task contracts; declarations only."""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from slimconfig import Output


class Tool(Enum):
    COMPUTE = "compute"


class Task(Enum):
    RUN = "build-folds"


@dataclass
class Fold(Output):
    train: Annotated[list[str], "Training sessions"]
    validation: Annotated[list[str], "Validation sessions"]
    test: Annotated[list[str], "Test sessions"]


@dataclass
class TaskInput(Output):
    sessions: Annotated[list[str], "Strictly chronological unique exchange session dates"]
    train_size: Annotated[int, "Rolling training window length"]
    validation_size: Annotated[int, "Validation window length"]
    test_size: Annotated[int, "Test window length and fold step"]
    label_horizon: Annotated[int, "Forward sessions consumed by each label"]
    embargo: Annotated[int, "Additional omitted sessions at each boundary"]


@dataclass
class TaskOutput(Output):
    folds: Annotated[list[Fold], "All complete rolling folds, omitting incomplete tails"]


TOOLS = {Tool.COMPUTE: (TaskInput, TaskOutput)}
TASKS = {Task.RUN: (TaskInput, TaskOutput)}
