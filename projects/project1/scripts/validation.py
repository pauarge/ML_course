import numpy as np

from helpers import compute_mse, build_poly, build_k_indices
from plots import cross_validation_visualization

"""
_, b_le = benchmark_lambda(lambda y, x, _: least_squares(y, x), ys_train, x_train, degree=7, plot_name="least_squares")
print(b_le[0])

_, b_r = benchmark_lambda(ridge_regression, ys_train, x_train, degree=7, plot_name="ridge_regression")
print(b_r)

_, b_le = benchmark_degrees(lambda y, x, _: least_squares(y, x), ys_train, x_train, plot_name="least_squares_degree")

_, b_le = benchmark_degrees(ridge_regression, ys_train, x_train, plot_name="ridge_regression_degree")
"""


def cross_validation(y, x, k_indices, k, f, lambda_, degree):
    """return the loss of ridge regression."""
    y_test = np.take(y, k_indices[k])
    x_test = np.take(x, k_indices[k])

    k_indices_new = np.delete(k_indices, k, 0)
    k_flattened_new = k_indices_new.flatten()

    y_train = np.take(y, k_flattened_new)
    x_train = np.take(x, k_flattened_new)

    tx_test = build_poly(x_test, degree)
    tx_train = build_poly(x_train, degree)

    w, _ = f(y_train, tx_train, lambda_)

    rmse_tr = np.sqrt(2 * compute_mse(y_train, tx_train, w))
    rmse_te = np.sqrt(2 * compute_mse(y_test, tx_test, w))
    return rmse_tr, rmse_te


def benchmark_lambda(f, ys_train, x_train, degree=1, plot_name="cross_validation"):
    seed = 1
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(ys_train, k_fold, seed)

    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    for i in lambdas:
        tr, te = 0, 0
        for j in range(k_fold):
            tmp_tr, tmp_te = \
                cross_validation(ys_train, x_train, k_indices, j, f, i, degree)
            tr += tmp_tr
            te += tmp_te
        rmse_tr.append(tr / k_fold)
        rmse_te.append(te / k_fold)

    cross_validation_visualization(lambdas, rmse_tr, rmse_te, degree, plot_name)

    return rmse_tr, rmse_te


def benchmark_degrees(f, ys_train, x_train, lambda_=0.01, plot_name="cross_validation"):
    seed = 1
    k_fold = 4
    degrees = range(1, 8)
    # split data in k fold
    k_indices = build_k_indices(ys_train, k_fold, seed)

    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    for i in degrees:
        tr, te = 0, 0
        for j in range(k_fold):
            tmp_tr, tmp_te = \
                cross_validation(ys_train, x_train, k_indices, j, f, lambda_, i)
            tr += tmp_tr
            te += tmp_te
        rmse_tr.append(tr / k_fold)
        rmse_te.append(te / k_fold)

    cross_validation_visualization(degrees, rmse_tr, rmse_te, lambda_, plot_name)

    return rmse_tr, rmse_te
