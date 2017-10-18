# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambda_ = lambda_ * 2 * y.shape[0]
    # return np.linalg.inv(np.transpose(tx).dot(tx) + lambda_*2*n * np.identity(tx.shape[1])).dot(np.transpose(tx).dot(y))
    return np.linalg.solve(np.transpose(tx).dot(tx) + np.multiply(lambda_, np.identity(tx.shape[1])),
                           np.transpose(tx).dot(y))
