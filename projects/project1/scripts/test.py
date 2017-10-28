import numpy as np
from datetime import datetime

from clean_data import look_for_999, standardize, standardize_train, remove_bad_data, remove_good_data, \
    discard_outliers, change_y_to_0
from helpers import predict_labels, compute_mse, build_k_indices, build_poly
from implementations import least_squares, least_squares_gd
from parsers import load_data, create_csv_submission
from validation import cross_validation, benchmark_degrees, benchmark_lambda, benchmark_outliers

OUT_DIR = "../out"


def main():
    ys_train, x_train, ids_train, x_test, ids_test = load_data()


    #x_train = remove_bad_data(x_train, ys_train)
    #x_test = remove_bad_data(x_test, _)
    #bad_columns = look_for_999(x_train)
    #x_train = np.delete(x_train,bad_columns,1)
    #x_test = np.delete(x_test, bad_columns, 1)


    print("FILTERING DATA")
    # proves per least_squares sense rows amb -999
    # x_train, ys_train = remove_bad_data(x_train, ys_train)


    # correlations = np.corrcoef(x_train1, rowvar=False)
    # np.fill_diagonal(correlations, 0.0)
    #
    # co = np.argmax(np.abs(correlations), axis=1)
    #
    # ws = dict()
    # for i in bad_columns:
    #     icc = co[i]
    #     tx_train = build_poly(x_train1[:, icc],1)
    #     w, _ = least_squares(x_train1[:, i], tx_train)
    #     ws[i] = w
    #
    #
    # subs = []
    # for i in bad_columns:
    #     subs.append((i,np.where(x_train[:,i] == -999)))
    #
    # for element, i in subs:
    #     col = element[0]
    #     w = ws[col]
    #     corr = co[i]
    #     for row in element[1]:
    #         x_train[row,col] = w[0] + w[1] * x_train[row,corr]


    #x_test, x_train = standardize(x_test, x_train)
    #x_train = standardize_train(x_train)

    #x_train, ys_train = discard_outliers(x_train, ys_train, 5)

    #tx_train = build_poly(x_train,3)

    #w, mse = least_squares(ys_train,tx_train)
    #print(mse)

    #w, mse = least_squares_gd(ys_train, tx_train, w, 1000, 0.0001)
    #print(mse)

    #rmse_te, rmse_tr = benchmark_degrees(ys_train, x_train,lambda_=0.01, plot_name="cross_validation")
    #print(rmse_te, rmse_tr)

    print("BUILDING POLYNOMIALS")
    # tx_train = build_poly(x_train, 3)
    # tx_test = build_poly(x_test, 3)


    # print("CHANGE Y")
    # ys_train = change_y_to_0(ys_train)

    # proves per least_squares sense rows amb -999
    # w, mse = least_squares(ys_train, tx_train)
    # print(mse)

    print("PREDICTING VALUES")
    # y_pred = predict_labels(w, tx_test)

    print("EXPORTING CSV")
    # create_csv_submission(ids_test, y_pred, "{}/submission-{}.csv".format(OUT_DIR, datetime.now()))


    rmse_tr, rmse_te = benchmark_degrees(ys_train, x_train, lambda_=0, plot_name="cross_validation LS degrees")
    print("TRAIN {}".format(rmse_tr))
    print("TRAIN {}".format(rmse_te))
    # benchmark_lambda(ys_train, x_train, degree=2, plot_name="PATATA_g2")
    # print(rmse_tr)
    # print(rmse_te)

    # seed = 3
    # k_fold = 4
    # # split data in k fold
    # k_indices = build_k_indices(ys_train, k_fold, seed)
    # loss_tr_l = []
    # loss_te_l = []
    # for k in range(k_fold):
    #     loss_tr, loss_te = cross_validation(ys_train, x_train, k_indices, k, 1, 11, lambda_=0)
    #     loss_te_l.append(loss_te)
    #     loss_tr_l.append(loss_tr)
    #
    # print(loss_tr_l)
    # print(loss_te_l)
    # print(np.mean(loss_tr_l))
    # print(np.mean(loss_te_l))


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
    for i in range(1,10):
        for j in range(k_fold):
            print(j)
            tr = 0
            te = 0
            tmp_tr, tmp_te = \
                cross_validation(ys_train, x_train, k_indices, j, i)
            tr += tmp_tr
            te += tmp_te
        rmse_tr.append(tr / k_fold)
        rmse_te.append(te / k_fold)

    print("TEST ERROR{}".format(rmse_te))
    print("TRAIN ERROR{}".format(rmse_tr))
    
    """


if __name__ == '__main__':
    main()
