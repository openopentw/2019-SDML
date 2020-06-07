""" Auto Encoder.  """

import torch
import torch.nn as nn

class Model(nn.Module):
    """ Classify genres. """
    def __init__(self, # pylint: disable=too-many-arguments
                 genre_size,
                 hid_dim,
                 drop_rate=0.0):

        super().__init__()

        # hidden
        self.hid = nn.Linear(genre_size, hid_dim)

        # relu
        self.acti = nn.LeakyReLU()

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
        genres = self.dropout(genres)
        # (batch_size, hid_dim)
        genres = torch.matmul(genres, self.hid.weight)
        genres = self.acti(genres)
        # (batch_size, genre_size)
        return genres
