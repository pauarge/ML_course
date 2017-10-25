from datetime import datetime
import numpy as np

from filters import discard_outliers, standardize, change_y_to_0
from helpers import predict_labels, compute_mse
from implementations import least_squares, build_poly, learning_by_gradient_descent, reg_logistic_regression
from parsers import load_data, create_csv_submission
from validation import benchmark_degrees

OUT_DIR = "../out"


def main():
    ys_train, x_train, ids_train, x_test, ids_test = load_data()

    print("FILTERING DATA")
    x_test, x_train = standardize(x_test, x_train)
    x_train, ys_train = discard_outliers(x_train, ys_train, 2)

    print("BUILDING POLYNOMIALS")
    tx_train = build_poly(x_train, 1)
    tx_test = build_poly(x_test, 1)

    print("CHANGE Y")
    ys_train = change_y_to_0(ys_train)

    print("PREDICTING VALUES")
    initial_w, _ = least_squares(ys_train,tx_train)
    w, losses = reg_logistic_regression(ys_train, tx_train, 0.01, initial_w, 100, 0.0001)
    print(w)
    print(losses)
    print(compute_mse(ys_train, tx_train, w))

    y_pred = predict_labels(w, tx_test)

    #print("Weight logistic reg:{}".format(w))
    #print(y_pred[0:10])

    print("EXPORTING CSV")
    create_csv_submission(ids_test, y_pred, "{}/submission-{}.csv".format(OUT_DIR, datetime.now()))


if __name__ == '__main__':
    main()
