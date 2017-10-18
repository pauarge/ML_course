# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""

import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(x.shape[0]))
    shuffled_y = y[shuffle_indices]
    shuffled_x = x[shuffle_indices]

    n = np.floor(x.shape[0] * ratio).astype(int)
    train_x, test_x = shuffled_x[:n], shuffled_x[n:]
    train_y, test_y = shuffled_y[:n], shuffled_y[n:]

    return train_x, test_x, train_y, test_y
