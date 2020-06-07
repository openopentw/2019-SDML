""" This model describes how to preict ratings.  """

import torch.nn as nn

class Model(nn.Module):
    """ Predict genres. """
    def __init__(self, config):
        super().__init__()

        (item_size,
         genre_size,
         hid_dim,
         drop_rate) = (config['args']['item_size'],
                       config['args']['genre_size'],
                       config['args']['hid_dim'],
                       config['args']['drop_rate'])

        # linear
        self.hid = nn.Linear(item_size, hid_dim)
        self.out = nn.Linear(hid_dim, genre_size)

        # non-linear
        self.acti = nn.ReLU()
        self.obj = nn.Sigmoid()

        # dropouts
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, item_item): # pylint: disable=arguments-differ
        """ Forward.
        Args:
            user: (batch_size,)
            item: (batch_size,)
        Returns:
            ratings: (batch_size,)
        """
        item = self.hid(item_item)
        item = self.acti(item)
        # (batch_size, hid_dim)
        item = self.out(item)
        item = self.obj(item)
        # (batch_size, genre_size)
        return item
