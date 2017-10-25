import numpy as np

from filters import discard_outliers, standardize, change_y_to_0
from helpers import predict_labels, compute_mse, build_k_indices
from implementations import least_squares, build_poly, learning_by_gradient_descent, logistic_regression
from parsers import load_data
from validation import cross_validation

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
    tx_test = build_poly(x_test, 1)

    # print("CHANGE Y")
    ys_train = change_y_to_0(ys_train)
    """
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

    print(loss)
    print(w)
    print(compute_mse(ys_train, tx_train, w))
    y_pred = predict_labels(w, tx_test)

    


    """
    seed = 1
    k_fold = 4
    k_indices = build_k_indices(ys_train, k_fold, seed)
    rmse_tr = []
    rmse_te = []
    tr = None
    te = None
    for j in range(k_fold):
        print(j)
        tr = 0
        te = 0
        tmp_tr, tmp_te, _ = \
            cross_validation(ys_train, x_train, k_indices, j, 1)
        tr += tmp_tr
        te += tmp_te
    rmse_tr.append(tr / k_fold)
    rmse_te.append(te / k_fold)

    print("TEST ERROR{}".format(rmse_te))
    print("TRAIN ERROR{}".format(rmse_tr))


if __name__ == '__main__':
    main()
