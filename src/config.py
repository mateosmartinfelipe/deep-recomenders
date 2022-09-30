from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class General:
    root: str


@dataclass
class ModelParamsConf:
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
class ModelParamsHpConf:
    dampping: List[float]
    lr: List[float]
    gamma: List[float]
    embbeding_size: List[float]


@dataclass
class ModelConfig:
    params: ModelParamsConf
    hp: Optional[ModelParamsHpConf]


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
