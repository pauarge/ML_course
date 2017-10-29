import numpy as np

from clean_data import standardize, discard_outliers, change_y_to_0, remove_bad_data, change_y_to_1
from helpers import compute_mse, build_poly, build_k_indices, learning_by_penalized_gradient, calculate_loss, \
    calculate_loss_reg
from implementations import least_squares, reg_logistic_regression, least_squares_gd
from plots import cross_validation_visualization, cross_validation_visualization_degree


def cross_validation(y, x, k_indices, k, degree, lambda_):
    """return the loss of ridge regression."""
    y_test = np.take(y, k_indices[k])
    x_test = np.take(x, k_indices[k], 0)

    k_indices_new = np.delete(k_indices, k, 0)
    k_flattened_new = k_indices_new.flatten()

    y_train = np.take(y, k_flattened_new)
    x_train = np.take(x, k_flattened_new, 0)

    y_train = change_y_to_0(y_train)
    y_test = change_y_to_0(y_test)

    tx_train = build_poly(x_train, degree)
    tx_test = build_poly(x_test, degree)

    # max_iters = 5000
    # gamma = 0.1
    # w_ini = np.ones(tx_train.shape[1])
    # w, mse_tr = least_squares_gd(y_train, tx_train, w_ini, max_iters, gamma)
    w, mse_tr = least_squares(y_train, tx_train)
    # w, loss_tr = reg_logistic_regression(y_train, tx_train, lambda_, w_ini, max_iters, gamma)

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
    return mse_tr, loss_te


def benchmark_lambda(ys_train, x_train, degree=1, plot_name="PATATA"):
    seed = 3
    k_fold = 4
    lambdas = np.logspace(-4, 0, 5)
    # split data in k fold
    k_indices = build_k_indices(ys_train, k_fold, seed)

    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    outliers = None

    for i in lambdas:
        print(i)
        tr, te = 0, 0
        for j in range(k_fold):
            tmp_tr, tmp_te = \
                cross_validation(ys_train, x_train, k_indices, j, degree, lambda_=i)
            tr += tmp_tr
            te += tmp_te
        print("Lambda {} Test Error {}".format(i, te / k_fold))

        rmse_tr.append(tr / k_fold)
        rmse_te.append(te / k_fold)

    cross_validation_visualization(lambdas, rmse_tr, rmse_te, degree, plot_name)

    return rmse_tr, rmse_te


def benchmark_degrees(ys_train, x_train, lambda_=0.01, plot_name="cross_validation"):
    seed = 3
    k_fold = 4
    degrees = range(1, 16)
    # split data in k fold
    k_indices = build_k_indices(ys_train, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    for i in degrees:
        print(i)
        tr, te = 0, 0
        for j in range(k_fold):
            tmp_tr, tmp_te = cross_validation(ys_train, x_train, k_indices, j, i, lambda_)
            # print(tmp_tr, j)
            # print(tmp_te, j)
            tr += tmp_tr
            te += tmp_te
        print("LOSS_TEST {} DEGREE {}".format(te / k_fold, i))
        rmse_tr.append(tr / k_fold)
        rmse_te.append(te / k_fold)

    cross_validation_visualization_degree(degrees, rmse_tr, rmse_te, lambda_, plot_name)
    # degree_min_te = min(enumerate(rmse_te), key=itemgetter(1))[0] + 1

    return rmse_tr, rmse_te


def ratio_of_acc(y_test, y_pred):
    print("CALCULATING SCORE")
    y_test = change_y_to_1(y_test)
    y_pred = change_y_to_1(y_pred)
    res = y_test * y_pred
    score = round((len(np.where(res > 0)[0])) / float(y_test.shape[0]), 10)
    print("Score: {}".format(score))
    return score
