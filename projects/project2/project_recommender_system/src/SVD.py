import math as mt
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

from methods import compute_error_SVD, global_mean, compute_std, standarize, div_std


def computesvd(train, K):
    mean = global_mean(train)
    train = standarize(train)
    std = compute_std(train)
    train = div_std(train)

    U, s, Vt = sp.linalg.svds(train, k=K)

    dim = (len(s), len(s))
    S = np.zeros(dim)
    for i in range(len(s)):
        S[i, i] = mt.sqrt(s[i])

    U = sp.lil_matrix(U, dtype=np.float32)  # dim(M,k)
    S = sp.lil_matrix(S, dtype=np.float32)  # dim(k,k)
    Vt = sp.lil_matrix(Vt, dtype=np.float32)  # dim(k,N)

    user_features = S.dot(Vt)
    item_features = np.transpose(U.dot(S))

    return user_features, item_features, mean, std


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
    print("COMPUTE SVD")
    user_features, item_features, mean, std = computesvd(train, num_features)
    print("GENERATE USER AND ITEM FEATURES")

    # # find the non-zero ratings indices
    # nz_row, nz_col = train.nonzero()
    # nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    # evaluate the test error
    rmse = compute_error_SVD(test, user_features, item_features, nz_test, mean, std)
    print("RMSE on test data: {}.".format(rmse))
    return item_features, user_features, rmse
