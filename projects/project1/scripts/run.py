from datetime import datetime
import numpy as np

from implementations import least_squares, build_poly, reg_logistic_regression, ridge_regression
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from helpers import standarize, dump_data, load_data
from validation import benchmark_lambda, benchmark_degrees


def main():
    print("PARSING TRAIN")
    ys_train, x_train, ids_train = load_data("ys_train"), load_data("x_train"), load_data("ids_train")
    if ys_train is None or x_train is None or ids_train is None:
        ys_train, x_train, ids_train = load_csv_data("../data/train.csv")
        dump_data(ys_train, "ys_train")
        dump_data(x_train, "x_train")
        dump_data(ids_train, "ids_train")

    print("PARSING TEST")
    x_test, ids_test = load_data("x_test"), load_data("ids_test")
    if x_test is None or ids_test is None:
        _, x_test, ids_test = load_csv_data("../data/test.csv")
        dump_data(x_test, "x_test")
        dump_data(ids_test, "ids_test")

    for i in range(x_test.shape[1]):
        x_test[:, i], x_train[:, i] = standarize(x_test[:, i], x_train[:, i])

    index = []
    for i in range(x_train.shape[0]):
        if np.amax(np.abs(x_train[i, :])) > 10:
            index.append(i)
    x_train = np.delete(x_train, index, 0)
    ys_train = np.delete(ys_train, index, 0)

    print("BUILDING POLY TRAIN")
    #tx_train = build_poly(x_train, 2)

    print("BUILDING POLY TEST")
    tx_test = build_poly(x_test, 2)

    #_, b_r = benchmark_lambda(ridge_regression, ys_train, x_train, degree=2, plot_name="ridge_regression")
    #print(b_r)

    for i in [0.0001, 0.001, 0.01, 0.1, 1]:
        _, b_le = benchmark_degrees(ridge_regression, ys_train, x_train, lambda_=i, plot_name="ridge_regression_degree_lambda{}".format[i])
        print(b_le)

    # w, loss = ridge_regression(ys_train, tx_train, 0.01)
    # print(len(w), loss)
    # y_pred = predict_labels(w, tx_test)

    #create_csv_submission(ids_test, y_pred, "../out/submission-{}.csv".format(datetime.now()))


if __name__ == '__main__':
    main()
