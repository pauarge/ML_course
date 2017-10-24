import numpy as np

from filters import discard_outliers, standardize, change_y
from helpers import predict_labels, compute_mse
from implementations import least_squares, build_poly, learning_by_gradient_descent

OUT_DIR = "../out"


def main():
    x_train = np.array([[5], [1], [3]], dtype=np.float64)
    ys_train = np.array([[0], [1], [0]], dtype=np.float64)

    x_test = np.array([[4], [5], [6]], dtype=np.float64)

    print("FILTERING DATA")
    x_test, x_train = standardize(x_test, x_train)
    x_train, ys_train = discard_outliers(x_train, ys_train, 2)

    print("BUILDING POLYNOMIALS")
    tx_train = build_poly(x_train, 1)
    tx_test = build_poly(x_test, 1)

    print("CHANGE Y")
    ys_train = change_y(ys_train)

    print("PREDICTING VALUES")
    w, mse = least_squares(ys_train, tx_train)
    print(mse)
    print("Weight least squares:{}".format(w))
    gamma = 0.0000001
    w, loss = learning_by_gradient_descent(ys_train, tx_train, w, gamma)
    print(loss)
    # print("Weight logistic reg:{}".format(w))
    y_pred = predict_labels(w, tx_test)

    print(compute_mse(ys_train, tx_train, w))


if __name__ == '__main__':
    main()
