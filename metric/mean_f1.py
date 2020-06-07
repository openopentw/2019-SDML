""" This module provide the f1 score. """

def mean_f1(pred, label, thres=0.5, eps=1e-8):
    """ Calculate F1-score.
    Args:
        pred: (batch_size, genre_size)
        label: (batch_size, genre_size)
    Return:
        f1: float
    """
    pred, label = (pred > thres), (label == 1)
    tp = (pred & label).sum(1).float() # pylint: disable=invalid-name
    fp = (pred & ~label).sum(1).float() # pylint: disable=invalid-name
    fn = (~pred & label).sum(1).float() # pylint: disable=invalid-name
    f1 = ((2 * tp) / (2 * tp + fp + fn + eps)).mean() # pylint: disable=invalid-name
    return f1
