import numpy as np

from clean_data import look_for_999
from filters import discard_outliers, standardize, change_y_to_0, remove_bad_data, remove_good_data
from helpers import predict_labels, compute_mse, build_k_indices, build_poly
from implementations import least_squares
from parsers import load_data
from validation import cross_validation, benchmark_degrees, benchmark_lambda

OUT_DIR = "../out"


def main():
    # x_train = np.array([[5], [1], [3]], dtype=np.float64)
    # ys_train = np.array([[0], [1], [0]], dtype=np.float64)

    # x_test = np.array([[4], [5], [6]], dtype=np.float64)

    ys_train, x_train, ids_train, x_test, ids_test = load_data()

    bad_columns = look_for_999(x_train)

    print("FILTERING DATA")
    # proves per least_squares sense rows amb -999
    x_train1, ys_train1 = remove_bad_data(x_train, ys_train)
    x_train2, ys_train2 = remove_good_data(x_train, ys_train)

    correlations = np.corrcoef(x_train1, rowvar=False)
    np.fill_diagonal(correlations, 0.0)

    co = np.argmax(np.abs(correlations), axis=1)

    ws = dict()
    for i in bad_columns:
        icc = co[i]
        tx_train = build_poly(x_train1[:, icc],1)
        w, _ = least_squares(x_train1[:, i], tx_train)
        ws[i] = w


    subs = []
    for i in bad_columns:
        subs.append((i,np.where(x_train[:,i] == -999)))

    for element, i in subs:
        col = element[0]
        w = ws[col]
        corr = co[i]
        for row in element[1]:
            x_train[row,col] = w[0] + w[1] * x_train[row,corr]


    # x_test, x_train1 = standardize(x_test, x_train1)
    # x_test2, x_train2 = standardize(x_test, x_train)

    x_train, ys_train = discard_outliers(x_train, ys_train, 1.95)

    print("BUILDING POLYNOMIALS")
    tx_train = build_poly(x_train, 3)
    tx_test = build_poly(x_test, 3)

    # print("CHANGE Y")
    # ys_train = change_y_to_0(ys_train)

    # proves per least_squares sense rows amb -999
    # w, mse = least_squares(ys_train, tx_train)
    # print(mse)

    # rmse_tr, rmse_te = benchmark_degrees(ys_train, x_train, lambda_=0, plot_name="cross_validation")
    # benchmark_lambda(ys_train, x_train, degree=2, plot_name="PATATA_g2")
    # print(rmse_tr)
    # print(rmse_te)

    seed = 56
    k_fold = 4
    # split data in k fold
    k_indices = build_k_indices(ys_train, k_fold, seed)
    loss_tr_l = []
    loss_te_l = []
    for k in range(k_fold):
        loss_tr, loss_te = cross_validation(ys_train, x_train, k_indices, k, 3, lambda_=0)
        loss_te_l.append(loss_te)
        loss_tr_l.append(loss_tr)

    print(loss_tr_l)
    print(loss_te_l)
    print(np.mean(loss_tr_l))
    print(np.mean(loss_te_l))
    """
    """
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
    """


if __name__ == '__main__':
    main()
