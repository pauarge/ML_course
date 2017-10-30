import numpy as np

from clean_data import standardize, discard_outliers, change_y_to_0, remove_bad_data, change_y_to_1
from helpers import compute_mse, build_poly, build_k_indices, learning_by_penalized_gradient, calculate_loss, \
    calculate_loss_reg
from implementations import least_squares, reg_logistic_regression, least_squares_gd
from plots import cross_validation_visualization, cross_validation_visualization_degree


def cross_validation(y, x, k_indices, k, degree, method, lambda_):
    """
    Cross validation using the specified method.
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param y: Vector of labels of size 1xN
    :param x: Matrix of input variables of size NxD
    :param k_indices: List of lists containing the indices of elements from x matrix that will be part of the test set
    :param k: Number of fold we are using for test set
    :param degree: Degree of the polynomial
    :param method: Method used to calculate w and error. It can be selected from Least Squares (LS) or
    Regularised Logistic Regression (RLR)
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

    # build polynomial of specified degree
    tx_train = build_poly(x_train, degree)
    tx_test = build_poly(x_test, degree)

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


def benchmark_lambda(ys_train, x_train, method, degree=1):
    """
    Cross validation using the specified method for different values of lambda_ and a fixed degree.
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param ys_train: Vector of labels of size 1xN
    :param x_train: Matrix of input variables of size NxD
    :param degree: Degree of the polynomial
    :param method: Method used to calculate w and error. It can be selected from Least Squares (LS) or
    Regularised Logistic Regression (RLR)
    :return: Returns two lists of errors, first one regarding train error and second one regarding test error, computed
    by either mean square (LS) or negative log likelihood (RLR); and the value of lambda_ corresponding to minimum test
    error value in the test.

    """
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

    cross_validation_visualization(lambdas, rmse_tr, rmse_te,
                                   plot_name="cross validation for lambdas with {}".format(method))

    min_lambda = lambdas[np.where(np.min(rmse_te))]

    return rmse_tr, rmse_te, min_lambda


def benchmark_degrees(ys_train, x_train, method, lambda_=0.01):
    """
    Cross validation using the specified method for different values of lambda_ and a fixed degree.
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param ys_train: Vector of labels of size 1xN
    :param x_train: Matrix of input variables of size NxD
    :param method: Method used to calculate w and error. It can be selected from Least Squares (LS) or
    Regularised Logistic Regression (RLR)
    :param lambda_: Regularization parameter
    :return: Returns two lists of errors, first one regarding train error and second one regarding test error,
    computed by either mean square (LS) or negative log likelihood (RLR); and the value of the degree corresponding
    to minimum test error value in the test.

    """
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

    cross_validation_visualization_degree(degrees, rmse_tr, rmse_te,
                                          plot_name="cross validation for degrees with {}".format(method))
    min_d = np.where(np.min(rmse_te)) + np.min(degrees)

    return rmse_tr, rmse_te, min_d


def ratio_of_acc(y_test, y_pred):
    """
    Calculates the percentage of correct predictions by comparing the real value with the predicted one.
    N = #data points

    :param y_test: Vector of real labels of size 1xN
    :param y_pred: Vector of predicted labels of size 1xN
    :return: Percentage of correct predictions

        """

    print("CALCULATING SCORE")
    y_test = change_y_to_1(y_test)
    y_pred = change_y_to_1(y_pred)
    res = y_test * y_pred
    score = 100*round((len(np.where(res > 0)[0])) / float(y_test.shape[0]), 10)
    print("Score: {}".format(score))
    return score
