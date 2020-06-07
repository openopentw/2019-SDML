""" This model describes how to preict ratings.  """

import torch
import torch.nn as nn

class Model(nn.Module):
    """ Predict genres. """
    def __init__(self, config):
        super().__init__()

        (user_size,
         item_size,
         embd_dim,
         drop_rate) = (config['args']['user_size'],
                       config['args']['item_size'],
                       config['args']['embd_dim'],
                       config['args']['drop_rate'])

        # embds
        self.embd_user = nn.Embedding(user_size, embd_dim)
        self.embd_user.weight.requires_grad = True
        self.embd_item = nn.Embedding(item_size, embd_dim)
        self.embd_item.weight.requires_grad = True

        # dropouts
        self.dropout = nn.Dropout(drop_rate)

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
        ratings = torch.bmm(user.unsqueeze(1), item.unsqueeze(2))
        # (batch_size,)
        return ratings.squeeze()
