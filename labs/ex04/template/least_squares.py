# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

from template.costs import compute_mse


def least_squares(y, tx):
    """calculate the least squares solution.
        returns mse, and optimal weights"""
    w = np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))
    # w = np.linalg.inv(np.transpose(tx).dot(tx)).dot(np.transpose(tx)).dot(y)
    mse = compute_mse(y, tx, w)
    return mse, w
