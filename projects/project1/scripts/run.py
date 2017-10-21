from datetime import datetime
import numpy as np

from implementations import least_squares, build_poly, ridge_regression
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from validation import benchmark_lambda, benchmark_degrees


def main():
    ys_train, x_train, ids_train = load_csv_data("../data/train.csv")
    _, x_test, ids_test = load_csv_data("../data/test.csv")

    """
    _, b_le = benchmark_lambda(lambda y, x, _: least_squares(y, x), ys_train, x_train, degree=7, plot_name="least_squares")
    print(b_le[0])

    _, b_r = benchmark_lambda(ridge_regression, ys_train, x_train, degree=7, plot_name="ridge_regression")
    print(b_r)

    _, b_le = benchmark_degrees(lambda y, x, _: least_squares(y, x), ys_train, x_train, plot_name="least_squares_degree")

    _, b_le = benchmark_degrees(ridge_regression, ys_train, x_train, plot_name="ridge_regression_degree")
    """

    tx_train = build_poly(x_train, 1)
    tx_test = build_poly(x_test, 1)
    w, _ = least_squares(ys_train, tx_train)
    y_pred = predict_labels(w, tx_test)

    create_csv_submission(ids_test, y_pred, "../out/submission-{}.csv".format(datetime.now()))


if __name__ == '__main__':
    main()
