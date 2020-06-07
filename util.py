""" Util functions """

import json
import random

import numpy as np
import torch

def init_seed(seed):
    """ Init some settings. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def make_label(genres_np):
    """ Transfer the labels into 0 1.
    Args:
        genres_np: np.array w. shape (27031,)
        label: np.array w. shape (27031, 17)
    Returns:
    """
    # collect all genres and find the max elm
    all_genres_list = []
    all_genres_list_flatten = []
    for elms in genres_np.tolist():
        if isinstance(elms, float):
            if np.isnan(elms):
                all_genres_list.append([])
            else:
                raise 'ERROR HERE'
        else:
            genres_list = [int(elm) for elm in elms.split()]
            all_genres_list.append(genres_list)
            all_genres_list_flatten += genres_list
    max_elm = max(all_genres_list_flatten) # 17
    # print(set(all_genres_list_flatten))
    # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}

    # create 0-1 table
    label = np.zeros((len(all_genres_list), max_elm + 1), int)
    for i, genres_list in enumerate(all_genres_list):
        for genres in genres_list:
            label[i, genres] = 1

    return label

def make_output(label):
    """ Transfer 0 1 labels back to indeces.
    Returns:
        output: List[str]
    """
    output = []
    for row in label:
        indeces = np.where(row == 1)[0]
        output.append(' '.join(str(idx) for idx in indeces))
    return output

def rand_drop(label, drop_rate=0.11):
    """ Randomly drop some genres. """
    dropped = np.array(label)
    mask_drop = np.random.rand(dropped.size) > (1 - drop_rate)
    mask_drop = mask_drop.reshape(*dropped.shape) # pylint: disable=not-an-iterable
    dropped *= (1 - mask_drop)
    return dropped, mask_drop

def find_best_hist(hist_path, verbose=True):
    """ Find best valid data in history. """
    hist = json.load(open(hist_path))
    valid_acc_list = [rec['valid']['acc'] for rec in hist]
    best_idx = valid_acc_list.index(max(valid_acc_list))
    if verbose:
        print('Best idx:', best_idx)
        print(hist[best_idx])
    return best_idx

def _dict_merge(a, b, path=None): # pylint: disable=invalid-name
    """ merges b into a.
    note that this mutates a - the contents of b are added to a
    (which is also returned).
    """
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                _dict_merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                # conflict: replace a[key] with b[key]
                a[key] = b[key]
                # raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def dict_merge(a, b): # pylint: disable=invalid-name
    """ merges b into a. """
    return _dict_merge(dict(a), b)
