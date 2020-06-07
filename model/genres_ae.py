""" Auto Encoder.  """

import torch.nn as nn

class Model(nn.Module):
    """ Classify genres. """
    def __init__(self, config):
        super().__init__()

        (genre_size,
         hid_dim,
         drop_rate) = (config['args']['genre_size'],
                       config['args']['hid_dim'],
                       config['args']['drop_rate'])

        # hidden
        self.hid = nn.Linear(genre_size, hid_dim)

        # classifier
        self.out = nn.Linear(hid_dim, genre_size)

        # relu
        self.acti = nn.ReLU()
        self.obj = nn.Sigmoid()

        # batch_norm
        self.batch_norm = nn.BatchNorm1d(hid_dim)

        # dropouts
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, genres): # pylint: disable=arguments-differ
        """ Forward.
        Args:
            genres: (batch_size, genres_ize)
        Returns:
            genres: (batch_size, genre_size)
        """
        genres = self.hid(genres)
        # genres = self.batch_norm(genres)
        # genres = self.acti(genres)
        genres = self.dropout(genres)
        # (batch_size, hid_dim)
        genres = self.out(genres)
        genres = self.acti(genres)
        # genres = self.obj(genres)
        # (batch_size, genre_size)
        return genres
