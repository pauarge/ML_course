from sparsesvd import sparsesvd
import math as mt
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
from scipy.sparse import csr_matrix

from methods import compute_error


def computesvd(train, K):
    print("compute SVD")
    U, s, Vt = sp.linalg.svds(train)

    dim = (len(s), len(s))
    S = np.zeros(dim)
    for i in range(len(s)):
        S[i, i] = mt.sqrt(s[i])

    U = sp.lil_matrix(np.transpose(U), dtype=np.float32)
    S = sp.lil_matrix(S, dtype=np.float32)
    Vt = sp.lil_matrix(Vt, dtype=np.float32)

    return U, S, Vt

def SGD(train, test, lambda_user, lambda_item, num_features):
    """matrix factorization by SGD."""
    # define parameters
    gamma = 0.01
    # num_features = 30  # K in the lecture notes
    # lambda_user = 0.1
    # lambda_item = 0.01
    num_epochs = 30  # number of full passes through the train set

    # set seed
    np.random.seed(988)

    # init matrix
    U, S, Vt = computesvd(train, num_features)
    user_features = S.dot(Vt)
    item_features = U

    # find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)

        # decrease step size
        gamma /= 1.2

        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            pred = user_info.T.dot(item_info)
            err = train[d, n] - pred[0,0]

            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)

        # rmse = compute_error(train, user_features, item_features, nz_train)
        # print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        print("iter: {}".format(it))

        # errors.append(rmse)

    # evaluate the test error
    rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(rmse))
    return item_features, user_features, rmse

