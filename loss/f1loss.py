""" This module provide the f1 loss. """

def f1_loss(pred, label, mask=0, mask_weight=3, eps=1e-8):
    """ Calculate F1-score.
    Args:
        pred: (batch_size, genre_size)
        label: (batch_size, genre_size)
    Return:
        f1: float
    """
    weight = mask * mask_weight + 1
    tp = (pred * label * weight).sum(1)
    fp = (pred * (1 - label) * weight).sum(1)
    fn = ((1 - pred) * label * weight).sum(1)
    f1 = ((2 * tp) / (2 * tp + fp + fn + eps))
    loss = 1 - f1
    return loss.mean()
