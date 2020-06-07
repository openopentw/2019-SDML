""" Auto Encoder.  """

import torch.nn as nn
from torchfm.model.fm import FactorizationMachineModel

class Model(nn.Module):
    """ Classify genres. """
    def __init__(self, # pylint: disable=too-many-arguments
                 genre_size,
                 hid_dim,
                 drop_rate=0.0):

        super().__init__()

        # hidden
        self.hid = nn.Linear(genre_size, hid_dim)
        self.fm = FactorizationMachineModel([genre_size], hid_dim) # pylint: disable=invalid-name

        # classifier
        self.out = nn.Linear(hid_dim, genre_size)

        # relu
        self.acti = nn.ReLU()

        # dropouts
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, genres): # pylint: disable=arguments-differ
        """ Forward.
        Args:
            genres: (batch_size, genres_ize)
        Returns:
            genres: (batch_size, genre_size)
        """
        genres = self.fm(genres)
        print(genres.shape)
        genres = self.dropout(genres)
        # (batch_size, hid_dim)
        genres = self.out(genres)
        genres = self.acti(genres)
        # (batch_size, genre_size)
        return genres
