""" This model describes how to preict ratings.  """

import torch
import torch.nn as nn

class Model(nn.Module):
    """ Predict genres. """
    def __init__(self, config):
        super().__init__()

        # embds
        self.embd_user = nn.Embedding(config.user_size, config.embd_dim)
        self.embd_user.weight.requires_grad = True
        self.embd_item = nn.Embedding(config.item_size, config.embd_dim)
        self.embd_item.weight.requires_grad = True

        # linear
        self.hid = nn.Linear(config.embd_dim * 2, config.hid_dim)
        self.out = nn.Linear(config.hid_dim, 1)

        # dropouts
        self.dropout = nn.Dropout(config.drop_rate)

    def forward(self, user, item): # pylint: disable=arguments-differ
        """ Forward.
        Args:
            user: (batch_size,)
            item: (batch_size,)
        Returns:
            ratings: (batch_size,)
        """
        user = self.embd_user(user)
        user = self.dropout(user)
        # (batch_size, embd_dim)
        item = self.embd_item(item)
        item = self.dropout(item)
        # (batch_size, embd_dim)
        ratings = torch.cat(user, item)
        # (batch_size, embd_dim * 2)
        ratings = self.hid(ratings)
        ratings = self.dropout(ratings)
        # (batch_size, hid_dim)
        ratings = self.out(ratings)
        # (batch_size,)
        return ratings
