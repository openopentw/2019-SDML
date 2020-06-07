""" This module trains the model. """

import importlib
from pathlib import Path

import torch.nn as nn
import torch.optim as optim

from trainer.trainer import Trainer as BaseTrainer

class Trainer(BaseTrainer):
    """ trainer. """
    def __init__(self, config, device):
        super().__init__()

        self.config = config

        # models
        model_src = 'model.{}'.format(Path(config['model']).stem)
        model_class = importlib.import_module(model_src).Model
        self.model = model_class(config).to(device)

        # init parameters
        self._init_model()

        # opt & loss
        self.opt = optim.Adam(self.model.parameters())
        self.loss = nn.MSELoss()

    def _run_batch(self, data, do_train):
        """ Train / Valid one batch. """
        user, item, ratings = data
        if do_train:
            self.opt.zero_grad()
        output_ratings = self.model(user, item)
        # (batch_size,)
        # calc loss
        loss = self.loss(output_ratings, ratings)
        if do_train:
            loss.backward()
            # opt
            self.opt.step()
        return loss.item(), loss.item()
