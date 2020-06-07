""" Auto Encoder.  """

import torch.nn as nn

class Model(nn.Module):
    """ Classify genres. """
    def __init__(self, # pylint: disable=too-many-arguments
                 genre_size,
                 hid_dim,
                 drop_rate=0.0):

        super().__init__()

        # sizes
        hid_dim_2 = int(hid_dim / 2)

        # hidden
        self.hid_0 = nn.Linear(genre_size, hid_dim)
        self.hid_1 = nn.Linear(hid_dim, hid_dim_2)
        self.hid_2 = nn.Linear(hid_dim_2, hid_dim)

        # classifier
        self.out = nn.Linear(hid_dim, genre_size)

        # non-linear
        self.acti = nn.ReLU()
        self.obj = nn.Sigmoid()

        # dropouts
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, genres): # pylint: disable=arguments-differ
        """ Forward.
        Args:
            genres: (batch_size, genres_ize)
        Returns:
            genres: (batch_size, genre_size)
        """
        genres_hid_0 = self.hid_0(genres)
        # (batch_size, hid_dim)
        genres = self.hid_1(genres_hid_0)
        genres = self.acti(genres)
        genres = self.dropout(genres)
        # (batch_size, hid_dim_2)
        genres = self.hid_2(genres)
        genres = self.acti(genres)
        genres += genres_hid_0 # residual connect
        # (batch_size, hid_dim)
        genres = self.out(genres)
        genres = self.acti(genres)
        # (batch_size, genre_size)
        return genres
