import logging
from pathlib import Path

import hydra
import onnx
import pandas as pd
import torch
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from config import ApplicationConf
from mlflow_utils import save_model, stage_to_prod
from models import Stats
from neural_collaborative_filtering import UserItemLabelSet, split_t_v_t, train

# Neutral Collaborative Filtering

logging.basicConfig(level=logging.DEBUG)


cs = ConfigStore.instance()
cs.store(name="application_conf", node=ApplicationConf)
# overwrite config arguments
# python src/main.py model_params.run_hp=False


@hydra.main(config_path="../conf", config_name="application.yaml")
def main(args: ApplicationConf):
    args.general.root = str(Path(__file__).parent.parent)
    # model output:
    models_folder = Path(args.general.root) / args.general.model_folder
    models_folder.mkdir(parents=True, exist_ok=True)
    # dataclass do not accept for  now Path objects
    items = Path(args.general.root) / args.data_params.dir / "PP_recipes.csv"
    logging.info(f"Items data : {items}")
    items = pd.read_csv(items)
    logging.info("Items :")
    logging.info(items.head())
    logging.info(items.columns)

    users = Path(args.general.root) / args.data_params.dir / "PP_users.csv"
    logging.info(f"Items data : {users}")
    users = pd.read_csv(users)
    logging.info("Items :")
    logging.info(users.head())
    logging.info(users.columns)

    iterations = (
        Path(args.general.root) / args.data_params.dir / "PP_users.csv"
    )
    logging.info(f"Items data : {iterations}")
    iterations = pd.read_csv(iterations).sample(frac=0.001)
    logging.info("Iterations :")
    logging.info(iterations.head())
    logging.info(iterations.columns)

    items_idx_mapping = {
        item: idx for idx, item in enumerate(items["id"].to_list())
    }
    user_idx_mapping = {
        user: idx for idx, user in enumerate(users["u"].to_list())
    }

    # DATA
    # users -> user_idx_mapping to create the Embedding for users
    # items -> items_idx_mapping to create the embedding for items
    # users_items -> (user,item) data to build the model
    train_data, validation_data, test_data = split_t_v_t(
        iterations,
        folder=Path(args.general.root) / "data",
        label=args.model_params.params.damping,
    )

    train_data = DataLoader(
        UserItemLabelSet(train_data),
        batch_size=args.model_params.params.batch_size,
        shuffle=True,
        num_workers=4,
    )
    validation_data = DataLoader(
        UserItemLabelSet(validation_data),
        batch_size=args.model_params.params.batch_size,
        shuffle=False,
    )
    test_data = DataLoader(
        UserItemLabelSet(test_data),
        batch_size=args.model_params.params.batch_size,
        shuffle=False,
    )
    # if args.model_params.run_hp:
    trained_model, final_loss = train(
        train=train_data,
        validate=validation_data,
        args=args.model_params,
        num_of_users=len(user_idx_mapping),
        num_of_items=len(items_idx_mapping),
        # the reason to detach those params from the args,
        # is to make it easier to test
        # pass the bo selection to a new model.
        embeddings_size=args.model_params.params.embedding_size,
        scheduler_step=args.model_params.params.step,
        scheduler_gamma=args.model_params.params.gamma,
        lr=args.model_params.params.lr,
    )
    stats = Stats(loss=final_loss)
    # saving the model as pt
    pt_file = Path(models_folder) / args.model_params.params.pt_name
    torch.save(trained_model.state_dict(), pt_file)
    # converting the model as onnx
    onnx_file = Path(models_folder) / args.model_params.params.onnx_name
    torch.onnx.export(
        trained_model, (torch.tensor([1]), torch.tensor([1])), onnx_file
    )
    # saving the models to MLFlow as Staging
    onnx_model = onnx.load(onnx_file)
    mlf_save_output = save_model(
        mlflow_uri=args.general.mlflow_url,
        model=onnx_model,
        experiment_name=args.model_params.params.model_root_name,
        model_name=args.model_params.params.onnx_name,
        stats=stats,
    )
    # Should go to Production
    productionalized = stage_to_prod(
        mlflow_uri=args.general.mlflow_url,
        stats=stats,
        model_name=args.model_params.params.onnx_name,
        train_model_info=mlf_save_output,
    )
    # making inference
    print(productionalized)


if __name__ == "__main__":
    main()
