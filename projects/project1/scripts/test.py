import numpy as np
from datetime import datetime

from clean_data import look_for_999, standardize, remove_bad_data, discard_outliers, change_y_to_0
from helpers import predict_labels, compute_mse, build_k_indices, build_poly
from implementations import least_squares, least_squares_gd
from parsers import load_data, create_csv_submission
from validation import cross_validation, benchmark_degrees, benchmark_lambda

OUT_DIR = "../out"


def main(argv, filter_data=False, benchmark=False, method=False):
    # load data
    ys_train, x_train, ids_train, x_test, ids_test = load_data()

    # determine way of pre-processing the information
    if filter_data:
        filter = argv[0]
    else:
        filter = 0

    # pre-process information
    if filter == 1:
        x_test, x_train = standardize(x_test, x_train)

    elif filter == 2:
        x_test, x_train = standardize(x_test, x_train)
        x_train, ys_train = discard_outliers(x_train, ys_train, 9)

    elif filter == 3:
        bad_columns = look_for_999(x_train)
        x_train = np.delete(x_train, bad_columns, 1)
        x_test = np.delete(x_test, bad_columns, 1)
        x_test, x_train = standardize(x_test, x_train)
        x_train, ys_train = discard_outliers(x_train, ys_train, 9)

    elif filter == 4:
        x_train, ys_train = remove_bad_data(x_train, ys_train)
        x_test = remove_bad_data(x_test, _)
        x_test, x_train = standardize(x_test, x_train)
        x_train, ys_train = discard_outliers(x_train, ys_train, 9)

    if method:
        method = arg[2]
    else:
        method = "LS"

    # 4-fold cross validation for different degrees
    if benchmark and argv[1] == "BD":
        loss_tr, loss_te, min_d = benchmark_degrees(ys_train, x_train, method, lambda_=0.01)
        print("MIN TEST ERROR: {} FOR {} DEGREE".format(np.min(loss_te), min_d))

    # 4-fold cross validation for different lambdas
    elif benchmark and arg[1] == "BL":
        loss_tr, loss_te, min_lambda = benchmark_lambda(ys_train, x_train, method, degree=1)
        print("MIN TEST ERROR: {} FOR LAMBDA {}".format(np.min(loss_te), min_lambda))


    # perform 4-fold cross validation with the specified method
    else:
        seed = 3
        k_fold = 4

        # split data in k fold
        k_indices = build_k_indices(ys_train, k_fold, seed)

        # specify degree
        degree = 4

        tr, te = 0, 0

        for j in range(k_fold):
            tmp_tr, tmp_te = \
                cross_validation(ys_train, x_train, k_indices, j, degree, method, lambda_=0.01)
            tr += tmp_tr
            te += tmp_te

        print("TEST ERROR {} FOR {} METHOD".format(te / k_fold, method))
        print("TRAIN ERROR {} FOR {} METHOD".format(tr / k_fold, method))


if __name__ == '__main__':
    main()
