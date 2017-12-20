import math as mt
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

from utils.methods import compute_error_SVD, global_mean, compute_std, standardize, div_std


def computesvd(train, K):
    """
    Computes SVD decomposition with scipy implementation
    :param train: Matrix of ratings from train set
    :param K: Number of features
    :return: Item and users features, mean and standard deviation
    """
    mean = global_mean(train)
    train = standardize(train, mean)
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


def SVD_SGD(train, test, lambda_user, lambda_item, num_features):
    """
    Update of SVD matrices
    :param train: Matrix of ratings from train set
    :param test: Matrix of ratings from test set
    :param lambda_user: Float value for the regularization term of the users
    :param lambda_item: Float value for the regularization term of the items
    :param num_features: Number of features
    :return: Item and users features, RMSE on test set.
    """

    # set seed
    np.random.seed(988)

    # init matrix
    print("COMPUTE SVD")
    user_features, item_features, mean, std = computesvd(train, num_features)
    pred = user_features.dot(item_features) * std + mean
    print("GENERATE USER AND ITEM FEATURES")

    # find the non-zero ratings indices
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    # evaluate the test error
    rmse = compute_error_SVD(test, user_features, item_features, nz_test, mean, std)
    print("RMSE on test data: {}.".format(rmse))
    return item_features, user_features, rmse
