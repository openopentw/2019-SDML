""" load data and preprocess and return dataloader. """

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from util import dict_merge

def get_dataloader(config, device):
    """ Get train / valid dataloader. """
    # read data
    ratings_df = pd.read_csv(config['ratings'])

    # sizes
    user_size = int(ratings_df.userId.max() + 1)
    item_size = int(ratings_df.itemId.max() + 1)

    # split
    (users,
     items,
     ratings) = (torch.from_numpy(ratings_df.userId.values).long().to(device),
                 torch.from_numpy(ratings_df.itemId.values).long().to(device),
                 torch.from_numpy(ratings_df.rating.values).float().to(device))
    dataset = TensorDataset(users, items, ratings)
    valid_size = int(len(dataset) * config['args']['valid'])
    (trainset,
     validset) = random_split(dataset, [len(dataset) - valid_size, valid_size])
    batch_size = config['args']['batch_size']
    (trainloader,
     validloader) = (DataLoader(trainset, batch_size=batch_size),
                     DataLoader(validset, batch_size=batch_size))

    # update config
    config['args'] = dict_merge(config['args'], {
        'user_size': user_size,
        'item_size': item_size,
    })

    return trainloader, validloader
