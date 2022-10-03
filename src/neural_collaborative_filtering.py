import itertools
import logging
import pickle
import random
from pathlib import Path
from typing import Callable, Iterable, List, Tuple, TypeVar

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import ModelConfig, ModelParamsHpConf

item_user = TypeVar("item_user", bound=Tuple[int, int, int])
record = TypeVar("record", bound=Tuple[int, int, int])


# this is a very naive way
def build_negative_sampling(
    items: Iterable[int], user_items: List[int], percent: float = 0.5
):
    available = items.difference(set(user_items))
    n = int(np.floor(len(user_items) * percent))
    negative = random.sample(available, n)
    return negative


def split_t_v_t(
    data: pd.DataFrame, folder: Path, label: int = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_file = folder / "train.pkl"
    validation_file = folder / "validation.pkl"
    test_file = folder / "test.pkl"
    if (
        train_file.is_file()
        and validation_file.is_file()
        and test_file.is_file()
    ):
        train_ = pickle.load(open(train_file, "rb"))
        validation_ = pickle.load(open(validation_file, "rb"))
        test_ = pickle.load(open(test_file, "rb"))
    else:
        train = 0.7
        validation = (1.0 - train) / 2
        test = 1 - train - validation
        users = data["u"].to_list()
        items_list = data["items"].to_list()
        train_ = []
        validation_ = []
        test_ = []
        items_list_as_list = [
            [int(n) for n in items_list[i].strip("][").split(", ")]
            for i in range(0, len(items_list))
        ]
        items_set = set(itertools.chain(*items_list_as_list))
        logging.info("CREATING POSITIVE AND NEGATIVE DATASET")
        for i, v in enumerate(tqdm(users)):
            for e in items_list_as_list[i]:
                r = random.random()
                if r <= test:
                    test_.append([v, e, label])
                elif test < random.random() <= test + validation:
                    validation_.append([v, e, label])
                else:
                    train_.append([v, e, label])

            negative_samples = build_negative_sampling(
                items=items_set, user_items=items_list_as_list[i], percent=0.5
            )
            for e in negative_samples:
                r = random.random()
                if r <= test:
                    test_.append([v, e, np.abs(label - 1.0)])
                elif test < random.random() <= test + validation:
                    validation_.append([v, e, np.abs(label - 1.0)])
                else:
                    train_.append([v, e, np.abs(label - 1.0)])
        train_ = np.asarray(train_)
        validation_ = np.array(validation_)
        test_ = np.array(test_)
        # we are convert the data to array otherwise
        # ( using list ) will give us a oom
        random.shuffle(train_)

        pickle.dump(train_, file=open(folder / "train.pkl", "wb"))
        pickle.dump(validation_, file=open(folder / "validation.pkl", "wb"))
        pickle.dump(test_, file=open(folder / "test.pkl", "wb"))
    return train_, validation_, test_


def set_up_model_objects(
    num_of_items_: int,
    num_of_users_: int,
    embeddings_size_: int,
    hidden_layers_dim_: List[int],
    lr: float,
    scheduler_step: int,
    scheduler_gamma: float,
):
    # WE NEED THE TRAINNG

    nfc = NCF(
        stage="training",
        num_of_items=num_of_items_,
        num_of_users=num_of_users_,
        embeddings_size=embeddings_size_,
        hidden_layers_dim=hidden_layers_dim_,
    )
    optimizer = torch.optim.Adam(params=nfc.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step,
        gamma=scheduler_gamma,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    return nfc, optimizer, scheduler, loss_fn


def training_step(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    records: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    # zero grads for batch ( torch accumulate grads)
    # CAVEATS MIGHT APPLY ( SET_TO_NONE )
    # Seems to have make the cpu used way better
    optimizer.zero_grad(set_to_none=True)
    # Try https://discuss.pytorch.org/t/dataloader-overrides-tensor-type/9861
    # but it did not work out , might be because
    # the output of the dataset is a tuple
    pred = model(records[:, 0].to(torch.int64), records[:, 1].to(torch.int64))
    # computing the loss and its fradients
    # loss = loss_fn(pred, elem[:, -1])
    loss = loss_fn(
        pred, torch.unsqueeze(records[:, -1].to(torch.float), dim=1)
    )
    loss.backward()
    # updating step
    optimizer.step()
    return loss


def train(
    train: DataLoader,
    validate: DataLoader,
    args: ModelConfig,
    num_of_items: int,
    num_of_users: int,
    embeddings_size: int,
    lr: float,
    scheduler_step: int,
    scheduler_gamma: float,
) -> Tuple[nn.Module, float]:
    # very difficult to test as objects are created inside the function.
    # on the other hand there is no need to write a test as the only thing
    # it does is run the loops
    model, optimizer, scheduler, loss_fn = set_up_model_objects(
        num_of_items_=num_of_items,
        num_of_users_=num_of_users,
        embeddings_size_=embeddings_size,
        hidden_layers_dim_=args.params.hidden_layers_dim,
        lr=lr,
        scheduler_step=scheduler_step,
        scheduler_gamma=scheduler_gamma,
    )
    running_loss = 0
    for epoch in range(0, args.params.epochs):
        for i, elem in enumerate(tqdm(train)):
            running_loss += training_step(
                optimizer=optimizer, model=model, records=elem, loss_fn=loss_fn
            )
            if i % 1000 == 0:
                last_lost = running_loss / 1000
                logging.info(
                    f"Train->  epoch :{epoch} batch :{i} loss :{last_lost}"
                )
                running_loss = 0

        running_loss = 0
        scheduler.step()
        with torch.inference_mode():
            running_loss = 0
            for i, elem in enumerate(validate):
                pred = model(
                    elem[:, 0].to(torch.int64), elem[:, 1].to(torch.int64)
                )
                loss = loss_fn(
                    pred, torch.unsqueeze(elem[:, -1].to(torch.float), dim=1)
                )
                running_loss += loss
            loss = running_loss / len(validate)
            logging.info(
                f"Validation ->  epoch :{epoch} batch :{i} loss :{loss}"
            )
        return model, loss


class NCF(nn.Module):
    def __init__(
        self,
        stage: str,
        num_of_items: int,
        num_of_users: int,
        embeddings_size: int,
        hidden_layers_dim: List[int],
        num_class: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # WETHER WE ARE TRAINING OR INFERRING
        self.stage = stage
        # LEARNABLE ITEM EMBEDDING
        self.items = nn.Embedding(num_of_items, embeddings_size)
        # LEARNABLE USER EMBEDDING
        self.users = nn.Embedding(num_of_users, embeddings_size)
        hidden_layers_dim.insert(0, embeddings_size * 2)
        self.norm = nn.LayerNorm(embeddings_size * 2)
        self.dense = nn.Linear(
            embeddings_size + hidden_layers_dim[-1], num_class
        )
        mlp_layers = []
        for units in range(1, len(hidden_layers_dim)):
            mlp_layers.append(
                nn.Linear(
                    hidden_layers_dim[units - 1],
                    hidden_layers_dim[units],
                    bias=True,
                )
            )
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.LayerNorm(hidden_layers_dim[units]))
            mlp_layers.append(nn.Dropout(dropout))
        self.mlp = nn.ModuleList(mlp_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        user_encode = self.users(user)
        item_encode = self.items(item)
        matrix_fact = torch.einsum("ij,ij -> ij", user_encode, item_encode)
        # matrix_fact = torch.matmul(user_encode, item_encode)
        user_item_tensor = self.norm(
            torch.cat([user_encode, item_encode], dim=1)
        )
        for layer in self.mlp:
            user_item_tensor = layer(user_item_tensor)
        out = self.dense(torch.cat([matrix_fact, user_item_tensor], dim=1))
        return out  # self.sigmoid(out)


# lets create a data Dataset class to deal with the data loading


class UserItemLabelSet(Dataset):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> item_user:

        return self.data[i]


class Optimitation:
    def __init__(self, hp_config: ModelParamsHpConf):
        ...
