""" This script do the baseline models. """

import argparse
import importlib
from pathlib import Path

import torch
import yaml

from util import init_seed, dict_merge

def parse_args():
    """ Parse args. """
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('config', help='config')
    parser.add_argument('--default_config', help='default config',
                        default='./config/default_args.yaml')

    args = parser.parse_args()
    return args

def run(): # pylint: disable=too-many-locals
    """ Run! """
    args = parse_args()

    # load config
    default_config = yaml.safe_load(open(args.default_config))
    config = yaml.safe_load(open(args.config))
    print('config loaded:', config)
    config = dict_merge(default_config, config)

    # init
    init_seed(config['args']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    preprocess_src = 'preprocess.{}'.format(Path(config['preprocess']).stem)
    preprocess_module = importlib.import_module(preprocess_src).get_dataloader
    (trainloader,
     validloader) = preprocess_module(config, device)

    # init trainer
    print('config:', config)
    trainer_src = 'trainer.{}'.format(Path(config['trainer']).stem)
    trainer_class = importlib.import_module(trainer_src).Trainer
    trainer = trainer_class(config, device)

    # train!
    model_dir = Path(config['model_dir'])
    if not model_dir.is_dir():
        model_dir.mkdir()
    trainer.train(trainloader,
                  validloader,
                  config['args']['epoch_size'],
                  model_dir)

if __name__ == '__main__':
    run()
