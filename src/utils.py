from typing import List, Tuple, TypeVar
import pandas as pd
import random
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np

item_user = TypeVar("item_user", bound=Tuple[int, int, int])
record = TypeVar("record", bound=Tuple[int, int, int])


def split_t_v_t(
    data: pd.DataFrame, label: int = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = 0.7
    validation = (1.0 - train) / 2
    test = 1 - train - validation
    users = data["u"].to_list()
    items_list = data["items"].to_list()
    train_ = []
    validation_ = []
    test_ = []

    for i, v in enumerate(users):
        items_view = [int(n) for n in items_list[i].strip("][").split(", ")]
        for e in items_view:
            r = random.random()
            if r <= test:
                test_.append([v, e, label])
            elif test < random.random() <= test + validation:
                validation_.append([v, e, label])
            else:
                train_.append([v, e, label])
    # we are convert the data to array otherwise ( using list ) will give us a oom
    random.shuffle(train_)
    train_ = np.asarray(train_)
    validation_ = np.array(validation_)
    test_ = np.array(test_)

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
