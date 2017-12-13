from sklearn.decomposition import NMF
from utils.parsers import load_data, create_submission
from utils.methods import compute_error_MF


def bucle():
    error_tr = []
    error_te = []
    for i in range(2, 21):
        r_tr, r_te = MF(i)
        error_tr.append(r_tr)
        error_te.append(r_te)
    print(error_tr)
    print(error_te)


def MF(num_feat):
    print("LOADING DATA...")
    train, test, transformation_user, transformation_item = load_data(1)
    print("STARTING MATRIX FACTORIZATION SGD")
    model = NMF(n_components=num_feat, init='random', solver='mu', random_state=0, max_iter=10000000)
    W = model.fit_transform(train)
    H = model.components_

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
