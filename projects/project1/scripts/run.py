from datetime import datetime
import numpy as np

from implementations import least_squares, build_poly, logistic_regression_newton, calculate_hessian
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission


def main():
    print("PARSING TRAIN")
    ys_train, x_train, ids_train = load_csv_data("../data/train.csv")

    print("PARSING TEST")
    _, x_test, ids_test = load_csv_data("../data/test.csv")

    print("BUILDING POLY TRAIN")
    tx_train = build_poly(x_train, 1)

    print("BUILDING POLY TEST")
    tx_test = build_poly(x_test, 1)

    w, _ = logistic_regression_newton(ys_train, tx_train)
    # y_pred = predict_labels(w, tx_test)

    # create_csv_submission(ids_test, y_pred, "../out/submission-{}.csv".format(datetime.now()))


if __name__ == '__main__':
    main()
