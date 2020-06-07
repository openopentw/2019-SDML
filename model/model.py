""" This model describes how to classify genres by viewing ratings as regularizer.  """

import torch
import torch.nn as nn

from model.genres import Genres
from model.ratings import Ratings

class Model(nn.Module):
    """ . """
    def __init__(self, # pylint: disable=too-many-arguments
                 user_size,
                 item_size,
                 emb_dim,
                 genre_size):

        super().__init__()

        # models
        self.model_genres = Genres(emb_dim, genre_size)
        self.model_ratings = Ratings()

        # embds
        self.embd_user = nn.Embedding(user_size, emb_dim)
        self.embd_user.weight.requires_grad = True
        self.embd_item = nn.Embedding(item_size, emb_dim)
        self.embd_item.weight.requires_grad = True

        # TODO: relu

        # TODO: dropouts

    def forward(self, user, item_1, item_2): # pylint: disable=arguments-differ
        """ Forward.
        Args:
            user: (batch_size,)
            item_1: (batch_size,)
            item_2: (batch_size,)
        Returns:
            ratings: (batch_size,)
            genres: (batch_size, genre_size)
        """
        user = self.embd_user(user)
        # (batch_size, emb_dim)
        item_1 = self.embd_item(item_1)
        # (batch_size, emb_dim)
        ratings = torch.bmm(user.squeeze(1), item_1.squeeze(2))
        # (batch_size)
        genres = self.out(item_2)
        # (batch_size, genre_size)
        return ratings, genres
