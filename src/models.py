from dataclasses import dataclass

from mlflow.entities import RunInfo


@dataclass
class Stats:
    loss: float


@dataclass
class MlFlowSaveModelOutput:
    run_info: RunInfo
    latest_version: int
    experiment_id: int
