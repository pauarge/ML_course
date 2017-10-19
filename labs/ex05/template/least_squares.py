# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    k = 1 / (2 * y.shape[0])
    e = y - tx.dot(w)
    return k * np.transpose(e).dot(e)


def least_squares(y, tx):
    """calculate the least squares solution.
        returns mse, and optimal weights"""
    w = np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))
    mse = compute_loss(y, tx, w)
    return mse, w
