from datetime import datetime

from clean_data import standardize, discard_outliers, change_y_to_0, remove_bad_data, look_for_999
from helpers import predict_labels, build_poly, predict_labels_log
from implementations import least_squares, reg_logistic_regression
from parsers import load_data, create_csv_submission
import numpy as np

OUT_DIR = "../out"


def main():
    ys_train, x_train, ids_train, x_test, ids_test = load_data()

    x_train, ys_train = remove_bad_data(x_train, ys_train)

    print("FILTERING DATA")
    x_test, x_train = standardize(x_test, x_train)
    x_train, ys_train = discard_outliers(x_train, ys_train, 10)

    # Just on logistic regression
    # ys_train = change_y_to_0(ys_train)

    print("BUILDING POLYNOMIALS")
    tx_train = build_poly(x_train, 11)
    tx_test = build_poly(x_test, 11)

    #print("LEARNING MODEL BY LEAST SQUARES")
    w, mse = least_squares(ys_train, tx_train)

    # print("LEARNING BY LOGISTIC REGRESSION")
    # lambda_ = 0
    # max_iters = 1000
    # gamma = 0.1
    # w, losses = reg_logistic_regression(ys_train, tx_train, lambda_, w_ini, max_iters, gamma)
    #
    # print("PREDICTING VALUES WITH LOG REGRESSION")
    # y_pred = predict_labels_log(w, tx_test)

    print("PREDICTING VALUES WITH LEAST SQ")
    y_pred = predict_labels(w, tx_test)

    print("EXPORTING CSV")
    create_csv_submission(ids_test, y_pred, "{}/submission-{}.csv".format(OUT_DIR, datetime.now()))


if __name__ == '__main__':
    main()
