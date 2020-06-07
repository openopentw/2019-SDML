""" This module trains the model. """

from abc import ABCMeta, abstractmethod
import json

import torch
from tqdm import tqdm, trange

class Trainer(metaclass=ABCMeta):
    """ trainer. """
    def __init__(self):
        self.config = None
        self.model = None
        self.opt = None
        self.loss = None

    def _init_model(self):
        """ Init some parameters. """
        # TODO: is this good?
        # init weights
        # def init_weights(model):
        #     for _, param in model.named_parameters():
        #         nn.init.uniform_(param.data, -0.08, 0.08)
        # self.model.apply(init_weights)

        # count parameters
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'The model has {count_parameters(self.model):,} trainable parameters')

    def train(self, trainloader, validloader, epoch_size, model_dir):
        """ Start training. """
        history = []
        for epoch_idx in trange(epoch_size, desc='Epoch'):
            train_loss, train_acc = self._run_epoch(trainloader, epoch_idx, True)
            valid_loss, valid_acc = self._run_epoch(validloader, epoch_idx, False)
            history.append({
                'train': {'loss': train_loss, 'acc': train_acc},
                'valid': {'loss': valid_loss, 'acc': valid_acc},
            })
            self._save_epoch(history, epoch_idx, model_dir)

    def _run_epoch(self, dataloader, epoch_idx, do_train):
        """ Train one epoch. """
        self.model.train(do_train)
        epoch_loss = 0
        epoch_acc = 0

        with torch.set_grad_enabled(do_train):
            with tqdm(enumerate(dataloader),
                      total=len(dataloader),
                      desc=('{} {}'.format(
                          'Train' if do_train else 'Valid',
                          epoch_idx
                      ))) as iter_:
                for batch_idx, data in iter_:
                    acc, loss = self._run_batch(data, do_train)
                    # sum & display
                    epoch_acc += acc
                    epoch_loss += loss
                    iter_.set_postfix(loss=epoch_loss / (batch_idx + 1),
                                      acc=epoch_acc / (batch_idx + 1))

        return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

    @abstractmethod
    def _run_batch(self, data, do_train):
        """ Train / Valid one batch.
        Returns:
            acc
            loss
        """

    def load(self, pkl_path):
        """ Load. """
        checkpoint = torch.load(pkl_path)
        self.model.load_state_dict(checkpoint['model'])
        self.opt.load_state_dict(checkpoint['optimizer'])

    def save(self, pkl_path):
        """ Save model. """
        self._save(pkl_path)

    def _save(self, pkl_path):
        """ Save model. """
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
        }
        torch.save(checkpoint, pkl_path)

    def _save_epoch(self, history, epoch_idx, model_dir):
        """ Save a epoch. """
        if not model_dir.is_dir():
            model_dir.mkdir()

        # save config
        with open(model_dir / 'config.json', 'w') as outfile:
            json.dump(self.config, outfile, indent=4)

        # save checkpoint
        self._save(model_dir / 'model_e-{}.pkl'.format(epoch_idx))

        # save history
        with open(model_dir / 'history.json', 'w') as outfile:
            json.dump(history, outfile, indent=4)
