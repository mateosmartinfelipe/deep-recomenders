from pathlib import Path
from typing import List, Tuple, TypeVar, Iterable
import pandas as pd
import random
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np
from copy import deepcopy
import itertools
import pickle

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
    if train_file.is_file() and validation_file.is_file() and test_file.is_file():
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
        for i, v in enumerate(users):
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
        # we are convert the data to array otherwise ( using list ) will give us a oom
        random.shuffle(train_)

        pickle.dump(train_, file=open(folder / "train.pkl", "wb"))
        pickle.dump(validation_, file=open(folder / "validation.pkl", "wb"))
        pickle.dump(test_, file=open(folder / "test.pkl", "wb"))
        return train_, validation_, test_


class NCF(nn.Module):
    def __init__(
        self,
        stage: str,
        num_of_items: int,
        num_of_users: int,
        embbeding_size: int,
        hidden_layers_dim: List[int],
        num_class: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # WETHER WE ARE TRAINING OR INFERING
        self.stage = stage
        # LEARNABLE ITEM EMBEDDING
        self.items = nn.Embedding(num_of_items, embbeding_size)
        # LERNABLE USER EMBEDDING
        self.users = nn.Embedding(num_of_users, embbeding_size)
        hidden_layers_dim.insert(0, embbeding_size * 2)
        self.norm = nn.LayerNorm(embbeding_size * 2)
        self.dense = nn.Linear(embbeding_size + hidden_layers_dim[-1], num_class)
        mlp_layers = []
        for units in range(1, len(hidden_layers_dim)):
            mlp_layers.append(
                nn.Linear(
                    hidden_layers_dim[units - 1], hidden_layers_dim[units], bias=True
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
        user_item_tensor = self.norm(torch.cat([user_encode, item_encode], dim=1))
        for layer in self.mlp:
            user_item_tensor = layer(user_item_tensor)
        out = self.dense(torch.cat([matrix_fact, user_item_tensor], dim=1))
        return self.sigmoid(out)


# lets create a data Dataset class to deal with the data loading


class UserItemLabelSet(Dataset):
    def __init__(self, data: np.ndarray) -> None:
        super(UserItemLabelSet, self).__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> item_user:

        return self.data[i]
