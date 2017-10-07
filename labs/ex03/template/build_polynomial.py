# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    tx = np.ones(x.shape)
    for i in range(1, degree + 1):
        tx = np.column_stack((tx, np.power(x, i)))
    return tx
