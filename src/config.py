from dataclasses import dataclass
from typing import List, Optional


@dataclass
class General:
    root: str


@dataclass
class Hp:
    type: str
    bounds: List[float]


@dataclass
class ModelParamsConf:
    dir: str
    epochs: int
    batch_size: int
    damping: float
    lr: float
    gamma: float
    step: float
    embedding_size: int
    hidden_layers_dim: List[int]


@dataclass
class ModelParamsHpConf:
    damping: Hp
    lr: Hp
    gamma: Hp
    embedding_size: Hp


@dataclass
class ModelConfig:
    params: ModelParamsConf
    hp: Optional[List[ModelParamsHpConf]]


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
