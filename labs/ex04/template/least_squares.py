# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def least_squares(y, tx):
    """calculate the least squares solution.
        returns mse, and optimal weights"""
    w = np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))
    # w = np.linalg.inv(np.transpose(tx).dot(tx)).dot(np.transpose(tx)).dot(y)
    mse = compute_mse(y, tx, w)
    return mse, w
