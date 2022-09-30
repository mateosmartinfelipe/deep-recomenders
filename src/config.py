from dataclasses import dataclass, field
from typing import List


@dataclass
class General:
    root: str


@dataclass
class ModelConfig:
    dir: str
    epochs: int
    batch_size: int
    dampping: float
    lr: float
    gamma: float
    step: float
    embbeding_size: int
    hidden_layers_dim: List[int]


@dataclass
class DataConfig:
    dir: str
    negative_percent: float
    train_percent: float
    train_dataset: str
    test_dataset: str
    validation_dataset: str


@dataclass
class ApplicationConf:
    general: General
    model_params: ModelConfig
    data_params: DataConfig
