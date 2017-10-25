import numpy as np

from filters import discard_outliers, standardize, change_y_to_0
from helpers import predict_labels, compute_mse
from implementations import least_squares, build_poly, learning_by_gradient_descent, logistic_regression
from parsers import load_data

OUT_DIR = "../out"


def main():
    # x_train = np.array([[5], [1], [3]], dtype=np.float64)
    # ys_train = np.array([[0], [1], [0]], dtype=np.float64)

    # x_test = np.array([[4], [5], [6]], dtype=np.float64)

    ys_train, x_train, ids_train, x_test, ids_test = load_data()

    print("FILTERING DATA")
    x_test, x_train = standardize(x_test, x_train)

    x_train, ys_train = discard_outliers(x_train, ys_train, 1.95)

    print("BUILDING POLYNOMIALS")
    tx_train = build_poly(x_train, 1)
    # tx_test = build_poly(x_test, 3)

    # print("CHANGE Y")
    ys_train = change_y_to_0(ys_train)

    print("PREDICTING VALUES")
    w = np.zeros(tx_train.shape[1])
    # print("Weight least squares:{}".format(w))
    gamma = 0.01
    for i in range(100):
        w, loss = learning_by_gradient_descent(ys_train, tx_train, w, gamma)
        print(i)
        if i % 50 == 0:
            print(loss)
            print(compute_mse(ys_train, tx_train, w))
            # print("Weight logistic reg:{}".format(w))
            # y_pred = predict_labels(w, tx_test)
    print(loss)
    print(w)
    print(compute_mse(ys_train, tx_train, w))


if __name__ == '__main__':
    main()
