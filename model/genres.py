""" This model describes how to classify genres.  """

import torch.nn as nn

class Genres(nn.Module):
    """ Classify genres. """
    def __init__(self, # pylint: disable=too-many-arguments
                 item_size,
                 genre_size,
                 embd_dim,
                 embd_pretrained=None,
                 embd_update=True,
                 drop_rate=0.5):

        super().__init__()

        self.embd_item = nn.Embedding(item_size, embd_dim)
        if embd_pretrained is not None:
            self.embd_item.weight = nn.Parameter(embd_pretrained)
        self.embd_item.weight.requires_grad = embd_update

        # classifier
        self.out = nn.Linear(embd_dim, genre_size)

        # relu
        self.acti = nn.ReLU()

        # dropouts
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, item): # pylint: disable=arguments-differ
        """ Forward.
        Args:
            item: (batch_size, embd_size)
        Returns:
            genres: (batch_size, genre_size)
        """
        item = self.embd_item(item)
        item = self.dropout(item)
        # (batch_size, embd_dim)
        genres = self.out(item)
        genres = self.acti(genres)
        # (batch_size, genre_size)
        return genres
