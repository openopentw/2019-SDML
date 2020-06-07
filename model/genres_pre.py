""" Auto Encoder.  """

import torch.nn as nn

class Model(nn.Module):
    """ Classify genres. """
    def __init__(self, config, embd_pretrained):

        super().__init__()

        # get sizes
        (item_size,
         genre_size,
         hid_dim,
         embd_dim,
         embd_update,
         drop_rate) = (config['args']['item_size'],
                       config['args']['genre_size'],
                       config['args']['hid_dim'],
                       config['args']['embd_dim'],
                       config['args']['embd_update'],
                       config['args']['drop_rate'])

        # embd
        self.embd_item = nn.Embedding(item_size, embd_dim)
        if embd_pretrained is not None:
            self.embd_item.weight = nn.Parameter(embd_pretrained)
        self.embd_item.weight.requires_grad = embd_update

        # linear
        self.hid = nn.Linear(genre_size, hid_dim)
        self.out = nn.Linear(hid_dim, genre_size)
        self.reg = nn.Linear(embd_dim, hid_dim)

        # non-linear
        self.acti = nn.ReLU()
        self.obj = nn.Sigmoid()

        # dropouts
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, item, genres): # pylint: disable=arguments-differ
        """ Forward.
        Args:
            item: (batch_size, item_size)
            genres: (batch_size, genres_ize)
        Returns:
            genres: (batch_size, genre_size)
        """
        ### auto encoder

        genres = self.hid(genres)
        genres = self.dropout(genres)
        out = self.acti(genres)
        # (batch_size, hid_dim)
        out = self.out(out)
        out = self.acti(out)
        # out = self.obj(out)
        # (batch_size, genre_size)

        ### regularized term

        item = self.embd_item(item)
        # (batch_size, embd_dim)
        item = self.reg(item)
        # (batch_size, hid_dim)
        reg = ((item - genres) ** 2).mean()
        # (batch_size,)

        return out, reg
