""" Auto Encoder.  """

import torch
import torch.nn as nn

class Model(nn.Module):
    """ Classify genres. """
    def __init__(self, # pylint: disable=too-many-arguments
                 genre_size,
                 hid_dim,
                 embd_dim,
                 drop_rate=0.0):

        super().__init__()
        self.eps = 1e-10

        # hidden
        self.hid = nn.Linear(embd_dim, hid_dim)

        # embds
        self.embd = nn.Embedding(genre_size, embd_dim)
        self.embd.weight.requires_grad = True

        # classifier
        self.out = nn.Linear(hid_dim, genre_size)

        # relu
        self.acti = nn.ReLU()

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
        interaction = torch.matmul(self.embd.weight, self.embd.weight.T)
        # (genre_size, genre_size)
        norm = (self.embd.weight ** 2).sum(1, True)
        norm = torch.matmul(norm, norm.T)
        # (genre_size, genre_size)
        interaction /= norm + self.eps
        # (genre_size, genre_size)
        attn_weight = torch.matmul(genres, interaction) / interaction.sum(1)
        # (batch_size, genre_size)
        genres = torch.matmul(attn_weight, self.embd.weight) / self.embd.weight.sum(0)
        genres = self.dropout(genres)
        # genres = self.batch_norm(genres)
        # (batch_size, embd_size)
        genres = self.hid(genres)
        genres = self.dropout(genres)
        # (batch_size, hid_dim)
        genres = self.out(genres)
        # genres = self.acti(genres)
        genres = torch.sigmoid(genres)
        # (batch_size, genre_size)
        return genres
