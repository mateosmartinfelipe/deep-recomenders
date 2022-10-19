from dataclasses import asdict

import mlflow
import onnx
from mlflow.entities import ViewType

from models import MlFlowSaveModelOutput, Stats


def save_model(
    mlflow_uri: str,
    model: onnx.TypeProto,
    experiment_name: str,
    model_name: str,
    stats: Stats,
) -> MlFlowSaveModelOutput:
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    experiment_id = mlflow.get_experiment_by_name(
        experiment_name
    ).experiment_id
    if experiment_id is None:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # logging stats for the trained model
        mlflow.log_metrics(asdict(stats))
        # Model search
        query = f"name='{model_name}'"
        register_model_ids = client.search_registered_models(
            filter_string=query  # , order_by=order_by
        )
        if not register_model_ids:
            client.create_registered_model(
                name=model_name,
                description="Deep Recommender",
            )

        model_uri = f"runs:/{run.info.run_id}/{model_name}"
        result = client.create_model_version(
            name=model_name, source=model_uri, run_id=run.info.run_id
        )
        latest_version = result.version

        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Staging",
        )
        mlflow.onnx.save_model(
            onnx_model=model,
            path=model_uri,
        )
        return MlFlowSaveModelOutput(run.info, latest_version, experiment_id)


def stage_to_prod(
    mlflow_uri: str,
    stats: Stats,
    model_name: str,
    train_model_info: MlFlowSaveModelOutput,
) -> bool:
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    # i need the staging model
    filter_string = f"name='{model_name}'"
    all_models = client.search_model_versions(filter_string)
    production_model = None
    for model in all_models:
        if model.current_stage == "Production":
            production_model = model
            break
    if production_model is None:
        client.transition_model_version_stage(
            name=model_name,
            version=train_model_info.latest_version,
            stage="Production",
        )
    else:
        # The default ordering is to sort by
        # ``start_time DESC``, then ``run_id``
        experiments = mlflow.search_runs(
            experiment_ids=[train_model_info.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            output_format="list",
        )
        production_model_stats = [
            Stats(**experiment.data.metrics)
            for experiment in experiments
            if experiment.info.run_id == production_model.run_id
        ]
        if stats.loss < production_model_stats[0].loss:
            client.transition_model_version_stage(
                name=model_name,
                version=production_model.version,
                stage="Archived",
            )

            client.transition_model_version_stage(
                name=model_name,
                version=train_model_info.latest_version,
                stage="Production",
            )
        else:
            print(
                f"""Recently trained model do not qualify to automatically transition ,
                 please have a look in case you want to transition manually.
                 Test Result:
                     Recently trained model loss: {stats.loss}
                     In prod model loss : {production_model_stats[0].loss}.
                 model name: {model_name}
                 Info {train_model_info.run_info}
                 """
            )
            return False
    return True
