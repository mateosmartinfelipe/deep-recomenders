from torch import nn
import torch
import os
from pathlib import Path, PurePath
import logging
import pandas as pd
from argparse import ArgumentParser
from typing import List
import random

# Neutral Collaborative Filtering
#
logging.basicConfig(level=logging.DEBUG)


def parse_args():
    parser = ArgumentParser(
        description="Train a Neural Collaborative" " Filtering model"
    )
    parser.add_argument("--root", type=str, help="project path")
    parser.set_defaults(root=PurePath(__file__).parent.parent)
    return parser.parse_args()


args = parse_args()


def main(args):
    items = PurePath(args.root) / "data" / "PP_recipes.csv"
    logging.info(f"Items data : {items}")
    items = pd.read_csv(items)
    logging.info(f"Items :")
    logging.info(items.head())
    logging.info(items.columns)

    users = PurePath(args.root) / "data" / "PP_users.csv"
    logging.info(f"Items data : {users}")
    users = pd.read_csv(users)
    logging.info(f"Items :")
    logging.info(users.head())
    logging.info(users.columns)

    iterations = PurePath(args.root) / "data" / "PP_users.csv"
    logging.info(f"Items data : {iterations}")
    iterations = pd.read_csv(iterations)
    logging.info(f"Iterations :")
    logging.info(iterations.head())
    logging.info(iterations.columns)

    items_idx_mapping = {item: idx for idx, item in enumerate(items["id"].to_list())}
    user_idx_mapping = {user: idx for idx, user in enumerate(users["u"].to_list())}
    # brute force as the length of the list varies
    users = iterations["u"].to_list()
    items_list = iterations["items"].to_list()
    users_items = []
    for i, v in enumerate(users):
        for e in [int(n) for n in items_list[i].strip("][").split(", ")]:
            users_items.append((v, e))
    random.shuffle(users_items)
    # DATA
    # users -> user_idx_mapping to create the Embeding for users
    # items -> items_idx_mapping to create the embedding for items
    # users_items -> (user,item) data to build the model


class NCF(nn.Module):
    def __init__(
        self,
        stage: str,
        num_of_items: int,
        num_of_users: int,
        embbeding_size: int,
        mlp_layers: List[int],
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
        num_layers = [embbeding_size * 2].extend(mlp_layers)
        self.norm = nn.LayerNorm(embbeding_size * 2)
        self.dense = nn.Linear(embbeding_size + num_layers[-1], num_class)
        mlp_layers = []
        for units in range(1, len(num_layers)):
            mlp_layers.append(
                nn.Linear(num_layers[units - 1], num_layers[units], bias=True)
            )
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.LayerNorm(num_layers[units]))
            mlp_layers.append(nn.Dropout(dropout))
        self.mlp = nn.ModuleList(mlp_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        user, item = input
        user_encode = self.users(user)
        item_encode = self.items(item)
        matrix_fact = torch.einsum("ij,ij -> ij")
        input = self.norm(torch.cat(user_encode, item_encode), axis=1)
        mlp_out = self.mlp(input)
        out = self.dense(torch.cat(matrix_fact, mlp_out))
        return self.sigmoid(out)

    optimizer = torch.optim.Adam()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss = torch.nn.MSELoss()


# WE NEED THE TRAINNG


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, help="project path")
    parser.set_defaults(root=PurePath(__file__).parent.parent)
    main(parser.parse_args())
