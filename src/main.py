import torch
from pathlib import Path
import logging
import pandas as pd
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import NCF, split_t_v_t, UserItemLabelSet
from hydra.core.config_store import ConfigStore
import hydra
from config import ApplicationConf

# Neutral Collaborative Filtering


logging.basicConfig(level=logging.DEBUG)


def parse_args():
    parser = ArgumentParser(
        description="Train a Neural Collaborative" " Filtering model"
    )
    parser.add_argument("--root", type=str, help="project path")
    return parser.parse_args()


cs = ConfigStore.instance()
cs.store(name="application_conf", node=ApplicationConf)


@hydra.main(config_path="../conf", config_name="application.yaml")
def main(args: ApplicationConf):
    cli_args = parse_args()
    if cli_args.root is not None:
        args.general.root = cli_args.root

    items = Path(args.general.root) / args.data_params.dir / "PP_recipes.csv"
    logging.info(f"Items data : {items}")
    items = pd.read_csv(items)
    logging.info(f"Items :")
    logging.info(items.head())
    logging.info(items.columns)

    users = Path(args.general.root) / args.data_params.dir / "PP_users.csv"
    logging.info(f"Items data : {users}")
    users = pd.read_csv(users)
    logging.info(f"Items :")
    logging.info(users.head())
    logging.info(users.columns)

    iterations = Path(args.general.root) / args.data_params.dir / "PP_users.csv"
    logging.info(f"Items data : {iterations}")
    iterations = pd.read_csv(iterations)
    logging.info(f"Iterations :")
    logging.info(iterations.head())
    logging.info(iterations.columns)

    items_idx_mapping = {item: idx for idx, item in enumerate(items["id"].to_list())}
    user_idx_mapping = {user: idx for idx, user in enumerate(users["u"].to_list())}

    # DATA
    # users -> user_idx_mapping to create the Embeding for users
    # items -> items_idx_mapping to create the embedding for items
    # users_items -> (user,item) data to build the model
    train_data, validation_data, test_data = split_t_v_t(
        iterations,
        folder=Path(args.general.root) / "data",
        label=args.model_params.dampping,
    )

    # WE NEED THE TRAINNG

    nfc = NCF(
        stage="training",
        num_of_items=len(items_idx_mapping),
        num_of_users=len(user_idx_mapping),
        embbeding_size=64,
        hidden_layers_dim=args.model_params.hidden_layers_dim,
    )
    optimizer = torch.optim.Adam(params=nfc.parameters(), lr=args.model_params.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.model_params.step, gamma=args.model_params.gamma
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train = DataLoader(
        UserItemLabelSet(train_data),
        batch_size=args.model_params.batch_size,
        shuffle=True,
        num_workers=4,
    )
    validation = DataLoader(
        UserItemLabelSet(validation_data),
        batch_size=args.model_params.batch_size,
        shuffle=False,
    )
    test = DataLoader(
        UserItemLabelSet(test_data),
        batch_size=args.model_params.batch_size,
        shuffle=False,
    )

    running_loss = 0
    for epoch in range(0, args.model_params.epochs):
        for i, elem in enumerate(tqdm(train)):
            # zero grads for batch ( torch accumulate grads)
            # CAVEATS MIGHT APPLY ( SET_TO_NONE ) # Seems to have make the cpu used way better
            optimizer.zero_grad(set_to_none=True)
            # Try https://discuss.pytorch.org/t/dataloader-overrides-tensor-type/9861
            # but it did not work out , might be becouse the output of the dataset is a tuple
            pred = nfc(elem[:, 0].to(torch.int64), elem[:, 1].to(torch.int64))
            # computing the loss and its fradients
            # loss = loss_fn(pred, elem[:, -1])
            loss = loss_fn(pred, torch.unsqueeze(elem[:, -1].to(torch.float), dim=1))
            loss.backward()
            # updating step
            optimizer.step()
            running_loss += loss
            if i % 1000 == 0:
                last_lost = running_loss / 1000
                logging.info(f"Train->  epoch :{epoch} batch :{i} loss :{last_lost}")
                running_loss = 0

        running_loss = 0
        scheduler.step()
        with torch.inference_mode():
            running_loss = 0
            for i, elem in enumerate(test):
                pred = nfc(elem[:, 0].to(torch.int64), elem[:, 1].to(torch.int64))
                loss = loss_fn(
                    pred, torch.unsqueeze(elem[:, -1].to(torch.float), dim=1)
                )
                running_loss += loss
            logging.info(
                f"Validation ->  epoch :{epoch} batch :{i} loss :{running_loss/len(test)}"
            )


if __name__ == "__main__":
    main()
