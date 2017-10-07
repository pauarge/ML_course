# -*- coding: utf-8 -*-
import numpy as np

"""Function used to compute the loss."""


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    k = 1 / (2 * y.shape[0])
    e = y - tx.dot(w)
    return k * np.transpose(e).dot(e)
