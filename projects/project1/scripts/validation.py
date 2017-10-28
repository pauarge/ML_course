import numpy as np

from clean_data import standardize, discard_outliers, change_y_to_0
from helpers import compute_mse, build_poly, build_k_indices, learning_by_penalized_gradient, calculate_loss
from implementations import least_squares, reg_logistic_regression
from plots import cross_validation_visualization


def cross_validation(y, x, k_indices, k, degree, outliers = 10, lambda_=0.00001):
    """return the loss of ridge regression."""
    y_test = np.take(y, k_indices[k])
    x_test = np.take(x, k_indices[k], 0)

    k_indices_new = np.delete(k_indices, k, 0)
    k_flattened_new = k_indices_new.flatten()

    y_train = np.take(y, k_flattened_new)
    x_train = np.take(x, k_flattened_new, 0)

    #x_test, x_train = standardize(x_test, x_train)
    #x_train, y_train = discard_outliers(x_train, y_train, outliers)

    y_train = change_y_to_0(y_train)
    y_test = change_y_to_0(y_test)

    tx_train = build_poly(x_train, degree)
    tx_test = build_poly(x_test, degree)


    #max_iters = 1000
    #gamma = 1
    w, mse_tr = least_squares(y_train, tx_train)
    #w, _ = reg_logistic_regression(y_train, tx_train, lambda_, w_ini, max_iters, gamma)


    loss_tr = mse_tr
    loss_te = compute_mse(y_test, tx_test, w)

    """
    #lambda_ = 0.00001
    max_iters = 5000
    gamma = 0.001
    #w, loss = learning_by_penalized_gradient(y_train, tx_train, lambda_, max_iters, gamma)
    w,loss = reg_logistic_regression(y_train, tx_train, lambda_, max_iters, gamma)

   
    
   
    loss_tr = calculate_loss(y_train, tx_train, w)
    loss_te = calculate_loss(y_test, tx_test, w)
    """
    return loss_tr, loss_te


def benchmark_lambda(ys_train, x_train, degree=1, plot_name="PATATA"):
    seed = 2
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(ys_train, k_fold, seed)

    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    outliers = None

    for i in lambdas:
        tr, te = 0, 0
        for j in range(k_fold):
            tmp_tr, tmp_te = \
                cross_validation(ys_train, x_train, k_indices, j, degree, outliers, lambda_=i)
            tr += tmp_tr
            te += tmp_te

        rmse_tr.append(tr / k_fold)
        rmse_te.append(te / k_fold)

    cross_validation_visualization(lambdas, rmse_tr, rmse_te, degree, plot_name)

    return rmse_tr, rmse_te


def benchmark_degrees(ys_train, x_train, lambda_=0.01, plot_name="cross_validation"):
    seed = 1
    k_fold = 4
    degrees = range(1, 15)
    # split data in k fold
    k_indices = build_k_indices(ys_train, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    for i in degrees:
        print(i)
        tr, te = 0, 0
        for j in range(k_fold):
            tmp_tr, tmp_te = cross_validation(ys_train, x_train, k_indices, j, i)
            # print(tmp_tr, j)
            # print(tmp_te, j)
            tr += tmp_tr
            te += tmp_te

        rmse_tr.append(tr / k_fold)
        rmse_te.append(te / k_fold)

    cross_validation_visualization(degrees, rmse_tr, rmse_te, lambda_, plot_name)
    # degree_min_te = min(enumerate(rmse_te), key=itemgetter(1))[0] + 1

    return rmse_tr, rmse_te


def benchmark_outliers(ys_train, x_train, lambda_=0.01, plot_name="cross_validation"):
    seed = 3
    k_fold = 4
    degrees = 1
    # split data in k fold
    k_indices = build_k_indices(ys_train, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    outliers = np.linspace(2, 15, 24)
    #outliers = [2]

    for i in outliers:
        print(i)
        tr, te = 0, 0
        for j in range(k_fold):
            tmp_tr, tmp_te = cross_validation(ys_train, x_train, k_indices, j, degrees, i)
            # print(tmp_tr, j)
            # print(tmp_te, j)
            tr += tmp_tr
            te += tmp_te

        rmse_tr.append(tr / k_fold)
        rmse_te.append(te / k_fold)
        print("OUT {}, TEST {}".format(i, rmse_te[-1]))

    cross_validation_visualization(degrees, rmse_tr, rmse_te, lambda_, plot_name)
    # degree_min_te = min(enumerate(rmse_te), key=itemgetter(1))[0] + 1

    return rmse_tr, rmse_te


def learning_by_log_reg(ys_train, tx_train, num_iter):
    w = np.zeros(tx_train.shape[1])
    # print("Weight least squares:{}".format(w))
    gamma = 0.01
    loss = None
    for i in range(num_iter):
        w, loss = learning_by_gradient_descent(ys_train, tx_train, w, gamma)
    return w, loss
