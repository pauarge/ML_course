from datetime import datetime

from clean_data import standardize, discard_outliers, change_y_to_0
from helpers import predict_labels, build_poly
from implementations import least_squares, reg_logistic_regression
from parsers import load_data, create_csv_submission

OUT_DIR = "../out"


def main():
    ys_train, x_train, ids_train, x_test, ids_test = load_data()

    print("FILTERING DATA")
    x_test, x_train = standardize(x_test, x_train)
    x_train, ys_train = discard_outliers(x_train, ys_train, 11)

    # Just on logistic regression
    # ys_train = change_y_to_0(ys_train)

    print("BUILDING POLYNOMIALS")
    tx_train = build_poly(x_train, 1)
    tx_test = build_poly(x_test, 1)

    print("LEARNING MODEL BY LEAST SQUARES")
    w, mse = least_squares(ys_train, tx_train)

    # print("LEARNING BY LOGISTIC REGRESSION")
    # lambda_ = 0
    # max_iters = 5000
    # gamma = 0.001
    # w, losses = reg_logistic_regression(ys_train, tx_train, lambda_, w, max_iters, gamma)

    print("PREDICTING VALUES")
    y_pred = predict_labels(w, tx_test)

    print("EXPORTING CSV")
    create_csv_submission(ids_test, y_pred, "{}/submission-{}.csv".format(OUT_DIR, datetime.now()))


if __name__ == '__main__':
    main()
