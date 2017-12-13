from sklearn.decomposition import NMF
from utils.parsers import load_data, create_submission
from utils.methods import compute_error_MF

def MF():

    print("LOADING DATA...")
    train, test, transformation_user, transformation_item = load_data(1)
    print("STARTING MATRIX FACTORIZATION SGD")
    model = NMF(n_components=20, init='random', solver='mu', random_state=0,max_iter=10000000)
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

    return W, H


if __name__ == '__main__':
    MF()
