import numpy as np

from clean_data import standardize, discard_outliers, change_y_to_0, remove_bad_data, change_y_to_1
from helpers import compute_mse, build_poly, build_k_indices, learning_by_penalized_gradient, calculate_loss, \
    calculate_loss_reg
from implementations import least_squares, reg_logistic_regression, least_squares_gd
from plots import cross_validation_visualization, cross_validation_visualization_degree


def cross_validation(y, x, k_indices, k, degree, method, lambda_):
    """return the train and test loss calculated with the specified method."""
    # define x and y test data set
    y_test = np.take(y, k_indices[k])
    x_test = np.take(x, k_indices[k], 0)

    k_indices_new = np.delete(k_indices, k, 0)
    k_flattened_new = k_indices_new.flatten()

    # define x and y train data set
    y_train = np.take(y, k_flattened_new)
    x_train = np.take(x, k_flattened_new, 0)

    # build polynomial of specified degree
    tx_train = build_poly(x_train, degree)
    tx_test = build_poly(x_test, degree)

    if method == "LS":
        #calculate w and loss by least squares method
        w, loss_tr = least_squares(y_train, tx_train)
        loss_te = compute_mse(y_test, tx_test, w)

    elif method == "RLR":
        # transform -1 y entries into 0s
        y_train = change_y_to_0(y_train)
        y_test = change_y_to_0(y_test)

        max_iters = 5000
        gamma = 0.1
        w_ini = np.ones(tx_train.shape[1])

        #calculate w and loss by regularized logistic regression
        w, loss_tr = reg_logistic_regression(y_train, tx_train, lambda_, w_ini, max_iters, gamma)
        loss_te = calculate_loss_reg(y_test, tx_test, w, lambda_)

    else:
        loss_tr = -999.0
        loss_te = -999.0
        print("SPECIFY METHOD")

    return loss_tr, loss_te


def benchmark_lambda(ys_train, x_train, method, degree=1):
    seed = 3
    k_fold = 4
    lambdas = np.logspace(-4, 0, 20)
    # split data in k fold
    k_indices = build_k_indices(ys_train, k_fold, seed)

    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    for i in lambdas:
        tr, te = 0, 0
        for j in range(k_fold):
            tmp_tr, tmp_te = \
                cross_validation(ys_train, x_train, k_indices, j, degree, method, lambda_=i)
            tr += tmp_tr
            te += tmp_te
        print("LOSS_TEST {} LAMBDA {}".format(te / k_fold, i))

        rmse_tr.append(tr / k_fold)
        rmse_te.append(te / k_fold)

    cross_validation_visualization(lambdas, rmse_tr, rmse_te, degree,
                                   plot_name = "cross validation for lambdas with {}".format(method))

    min_lambda = lambdas[np.where(np.min(rmse_te))]

    return rmse_tr, rmse_te, min_lambda


def benchmark_degrees(ys_train, x_train, method, lambda_=0.01):
    seed = 1
    k_fold = 4
    degrees = range(1, 16)
    # split data in k fold
    k_indices = build_k_indices(ys_train, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    for i in degrees:
        tr, te = 0, 0
        for j in range(k_fold):
            tmp_tr, tmp_te = cross_validation(ys_train, x_train, k_indices, j, i, method, lambda_)
            tr += tmp_tr
            te += tmp_te

        print("LOSS_TEST {} DEGREE {}".format(te / k_fold, i))
        rmse_tr.append(tr / k_fold)
        rmse_te.append(te / k_fold)

    cross_validation_visualization_degree(degrees, rmse_tr, rmse_te, lambda_,
                                          plot_name = "cross validation for degrees with {}".format(method))
    min_d = np.where(np.min(rmse_te)) + np.min(degrees)

    return rmse_tr, rmse_te, min_d


def ratio_of_acc(y_test, y_pred):
    print("CALCULATING SCORE")
    y_test = change_y_to_1(y_test)
    y_pred = change_y_to_1(y_pred)
    res = y_test * y_pred
    score = round((len(np.where(res > 0)[0])) / float(y_test.shape[0]), 10)
    print("Score: {}".format(score))
    return score
