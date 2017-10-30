import numpy as np
import argparse

from clean_data import look_for_999, standardize, remove_bad_data, discard_outliers
from helpers import build_k_indices
from parsers import load_data
from validation import cross_validation, benchmark_degrees, benchmark_lambda

OUT_DIR = "../out"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=int, help='choose the pre-processing data way\n0: raw data\n1: standardize\n'
                                                   '2: standardize + discard outliers\n'
                                                   '3: remove features with -999 values\n'
                                                   '4: remove data points with -999 values', default=2)
    parser.add_argument('--method', type=str, help="choose between Least Squares ('LS') or Regularized Logistic "
                                                   "Regression ('RLR')", default="LS")
    parser.add_argument('--X', type=str, help="choose between X-validation among different degrees ('BD') or "
                                              "lambdas ('BL'). For simple X-validation with the default hyperparameters"
                                              "setting, choose 'XV'", default="BD")
    args = parser.parse_args()


    ys_train, x_train, ids_train, x_test, ids_test = load_data()

    # pre-processing the information
    if args.filter == 1:
        x_test, x_train = standardize(x_test, x_train)

    elif args.filter == 2:
        x_test, x_train = standardize(x_test, x_train)
        x_train, ys_train = discard_outliers(x_train, ys_train, 9)

    elif args.filter == 3:
        bad_columns = look_for_999(x_train)
        x_train = np.delete(x_train, bad_columns, 1)
        x_test = np.delete(x_test, bad_columns, 1)
        x_test, x_train = standardize(x_test, x_train)
        x_train, ys_train = discard_outliers(x_train, ys_train, 9)

    elif args.filter == 4:
        x_train, ys_train = remove_bad_data(x_train, ys_train)
        x_test = remove_bad_data(x_test, _)
        x_test, x_train = standardize(x_test, x_train)
        x_train, ys_train = discard_outliers(x_train, ys_train, 9)


    method = args.method

    # 4-fold cross validation for different degrees
    if args.X == "BD":
        loss_tr, loss_te, min_d = benchmark_degrees(ys_train, x_train, method, lambda_=0.01)
        print("MIN TEST ERROR: {} FOR {} DEGREE".format(np.min(loss_te), min_d))

    # 4-fold cross validation for different lambdas
    elif args.X == "BL":
        loss_tr, loss_te, min_lambda = benchmark_lambda(ys_train, x_train, method, degree=1)
        print("MIN TEST ERROR: {} FOR LAMBDA {}".format(np.min(loss_te), min_lambda))

    # perform 4-fold cross validation with the specified method
    else:
        seed = 3
        k_fold = 4

        # split data in k fold
        k_indices = build_k_indices(ys_train, k_fold, seed)

        # specify degree
        degree = 11

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
