from sklearn.decomposition import NMF
import numpy as np
from utils.parsers import load_data, create_submission
from utils.methods import compute_error_MF, global_mean, standarize, compute_std, div_std, compute_error_bias


def bucle():
    error_tr = []
    error_te = []
    for i in range(10, 15):
        for j in [0.001]:
            r_tr, r_te = MF(i, j)
            error_tr.append(r_tr)
            error_te.append(r_te)
    print(error_tr)
    print(error_te)


def MF(num_feat, alp):
    train, test, _, _ = load_data()

    mean = global_mean(train)
    # train = standarize(train, mean)
    std = compute_std(train)
    train = div_std(train)

    model = NMF(n_components=num_feat, init='nndsvda', solver='mu', random_state=0, max_iter=10000000, alpha=alp,
                verbose=1)
    W = model.fit_transform(train)
    H = model.components_
    pred = (W.dot(H) * std)
    pred += mean * np.ones((pred.shape[0], pred.shape[1]))

    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    rmse_train = compute_error_MF(train, H, W, nz_train)
    rmse_test = compute_error_MF(test, H, W, nz_test)

    print("RMSE on train data: {}.".format(rmse_train))
    print("RMSE on test data: {}.".format(rmse_test))

    return rmse_train, rmse_test


if __name__ == '__main__':
    bucle()
