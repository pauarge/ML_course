import numpy as np

from clean_data import standardize, discard_outliers, change_y_to_0, remove_bad_data, change_y_to_1
from helpers import compute_mse, build_poly, build_k_indices, learning_by_penalized_gradient, calculate_loss, \
    calculate_loss_reg
from implementations import least_squares, reg_logistic_regression, least_squares_gd
from plots import cross_validation_visualization_degree, cross_validation_visualization_lambdas


def cross_validation(y, x, k_indices, k, lambda_):
    """
    Cross validation using the specified method.
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param y: Vector of labels of size 1xN
    :param x: Matrix of input variables of size NxD
    :param k_indices: List of lists containing the indices of elements from x matrix that will be part of the test set
    :param k: Number of fold we are using for test set
    :param lambda_: Regularization parameter
    :return: error of the weights computed by either mean square (LS) or negative log likelihood (RLR) for both train
    and test sets

    """
    # define x and y test data set
    y_test = np.take(y, k_indices[k])
    x_test = np.take(x, k_indices[k], 0)

    k_indices_new = np.delete(k_indices, k, 0)
    k_flattened_new = k_indices_new.flatten()

    # define x and y train data set
    y_train = np.take(y, k_flattened_new)
    x_train = np.take(x, k_flattened_new, 0)

    if method == "LS":
        # calculate w and loss by least squares method
        w, loss_tr = least_squares(y_train, tx_train)
        loss_te = compute_mse(y_test, tx_test, w)

    elif method == "RLR":
        # transform -1 y entries into 0s
        y_train = change_y_to_0(y_train)
        y_test = change_y_to_0(y_test)

        max_iters = 5000
        gamma = 0.1
        w_ini = np.ones(tx_train.shape[1])

        # calculate w and loss by regularized logistic regression
        w, loss_tr = reg_logistic_regression(y_train, tx_train, lambda_, w_ini, max_iters, gamma)
        loss_te = calculate_loss_reg(y_test, tx_test, w, lambda_)

    else:
        loss_tr = -999.0
        loss_te = -999.0
        print("SPECIFY METHOD")

    return loss_tr, loss_te
