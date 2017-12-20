import numpy as np
import scipy as sp
from sklearn.decomposition import NMF

from utils.helpers import build_index_groups


def init_mf(train, num_features):
    """
    Initialize the parameters for matrix factorization.
    :param train: Matrix of ratings from train set
    :param num_features: Integer of number of features to generate the matrices
    :return: User and Item matrices of features
    """

    num_item, num_user = train.get_shape()

    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)

    # start by item features.
    item_nnz = train.getnnz(axis=1)
    item_sum = np.array(train.sum(axis=1))

    print("GENERATING SPECIAL FEATURE")
    for ind in range(num_item):
        if item_nnz[ind] != 0:
            item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]

    print(user_features.shape, item_features.shape)
    return user_features, item_features


def compute_error(data, user_features, item_features, nz):
    """
    Compute the loss (RMSE) of the prediction of nonzero elements.
    :param data: Matrix of ratings
    :param user_features: Matrix of user features
    :param item_features: Matrix of item features
    :param nz: List of indexes of non-zero elements of the data
    :return: RMSE loss
    """
    mse = 0
    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        mse += (data[row, col] - user_info.T.dot(item_info)) ** 2
    return np.sqrt(1.0 * mse / len(nz))


def compute_error_SVD(data, user_features, item_features, nz, mean, std):
    """
    Compute the loss (RMSE) of the prediction of nonzero elements when using SVD method.
    :param data: Matrix of ratings
    :param user_features: Matrix of user features
    :param item_features: Matrix of item features
    :param nz: List of indexes of non-zero elements of the data
    :param mean: Global mean of the non-zero ratings in the train set
    :param std: Global std of the non-zero ratings in the train set
    :return: Matrix of predictions and RMSE
    """
    mse = 0
    pred = item_features.T.dot(user_features)

    error = []
    for row, col in nz:
        error_pred = (data[row, col] - (std * pred[row, col] + mean)) ** 2
        mse += error_pred
        error.append(error_pred)

    return pred, np.sqrt(1.0 * mse / len(nz))


def decomposition_error(ratings, data, user_features, item_features, nz, mean, std, user_bias, item_bias, min_num):
    """
    Generates the decomposition of the error to study the accurancy of the predictions in
    different type of users and items.
    :param ratings:
    :param data:
    :param user_features:
    :param item_features:
    :param nz:
    :param mean:
    :param std:
    :param user_bias:
    :param item_bias:
    :param min_num:
    :return:
    """
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    bad_users = num_items_per_user < min_num
    bad_items = num_users_per_item < min_num

    mse_1 = 0
    i1 = 0
    mse_2 = 0
    i2 = 0
    mse_3 = 0
    i3 = 0
    mse_4 = 0
    i4 = 0

    prediction = np.transpose(item_features.T.dot(user_features))

    for row, col in nz:
        rate = std * prediction[row, col] + mean + user_bias[col] + item_bias[row]
        if bad_users[col]:
            if bad_items[row]:
                mse_4 += (data[row, col] - rate) ** 2
                i4 += 1
            else:
                mse_2 += (data[row, col] - rate) ** 2
                i2 += 1
        elif bad_items[row]:
            mse_3 += (data[row, col] - rate) ** 2
            i3 += 1
        else:
            mse_1 += (data[row, col] - rate) ** 2
            i1 += 1
            if i1 % 10000 == 0:
                print(mse_1, mse_2, mse_3, mse_4)

    mse = [mse_1, mse_2, mse_3, mse_4]
    percentage = mse / sum(mse)
    ii = [i1, i2, i3, i4]
    for i in range(4):
        if ii[i] != 0:
            mse[i] = np.math.sqrt(mse[i] / ii[i])
        else:
            mse[i] = 0

    return mse[0], mse[1], mse[2], mse[3], percentage, np.math.sqrt(sum(mse) / len(nz))


def matrix_factorization_SGD(train, test, lambda_user, lambda_item, num_features):
    """
    Matrix factorization by SGD.
    :param train: Matrix of ratings for train set
    :param test: Matrix of ratings for test set
    :param lambda_user: Regularization parameter for user's matrix
    :param lambda_item: Regularization parameter for item's matrix
    :param num_features: Number of features
    :return: Item and user's matrices and RMSE on test set.
    """
    # define parameters
    gamma = 0.01
    num_epochs = 50  # number of full passes through the train set

    # set seed
    np.random.seed(1)

    # init matrix
    print("INITIALIZE MATRICES...")
    user_features, item_features = init_mf(train, num_features)

    # find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    print("LEARN MATRIX FACTORIZATION USING SGD...")
    for it in range(num_epochs):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)

        # decrease step size
        gamma /= 1.2

        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            err = train[d, n] - user_info.T.dot(item_info)

            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)

        if it % 5 == 0:
            rmse = compute_error(train, user_features, item_features, nz_train)
            print("iter: {}, RMSE on training set: {}.".format(it, rmse))

    # evaluate the test error
    rmse_te = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(rmse_te))
    return item_features, user_features, rmse_te


def matrix_factorization_sgd_std(train, test, lambda_user, lambda_item, num_features, u_bias, i_bias):
    """
    Matrix factorization by SGD.
    :param train: Matrix of ratings for train set
    :param test: Matrix of ratings for test set
    :param lambda_user: Float value for regularization parameter for user's matrix
    :param lambda_item: Float value for regularization parameter for item's matrix
    :param num_features: Number of features
    :param u_bias: List of users bias
    :param i_bias: List of items bias
    :return: Item and user's matrices and RMSE on test set.
    """
    # define parameters
    gamma = 0.01
    num_epochs = 50  # number of full passes through the train set

    # set seed
    np.random.seed(988)

    # init matrix
    print("INITIALIZE MATRICES...")
    user_features, item_features = init_mf(train, num_features)

    # find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    print("LEARN MATRIX FACTORIZATION USING SGD...")
    for it in range(num_epochs):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)

        # decrease step size
        gamma /= 1.2

        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            err = train[d, n] - (user_info.T.dot(item_info) + u_bias[n] + i_bias[d])

            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)

        if it % 5 == 0:
            rmse_te = compute_error(train, user_features, item_features, nz_train)
            print("iter: {}, RMSE on training set: {}.".format(it, rmse_te))

    # evaluate the test error
    rmse_te = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(rmse_te))
    return item_features, user_features, rmse_te


def matrix_factorization_sk(train, test, num_feat=2, alp=0.01):
    """
    Matrix factorization implementation with SKlearn library
    :param train: Matrix of ratings for train set
    :param test: Matrix of ratings for test set
    :param num_feat: Number of features
    :param alp: Learning rate
    :return: User and item matrices, RMSE on test set.
    """
    model = NMF(n_components=num_feat, init='nndsvda', solver='mu', random_state=0, max_iter=10000000, alpha=alp,
                verbose=1)
    W = model.fit_transform(train)
    H = model.components_

    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    rmse_train = compute_error(train, H, W, nz_train)
    rmse_test = compute_error(test, H, W, nz_test)

    print("RMSE on train data: {}.".format(rmse_train))
    print("RMSE on test data: {}.".format(rmse_test))

    return W, H, rmse_test


# ALTERNATING LEAST SQUARES

def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """
    Update user feature matrix.
    :param train: Matrix of ratings from train set
    :param item_features: Matrix of item features
    :param lambda_user: Float value of regularization parameter for user's matrix
    :param nnz_items_per_user: List of non-zero elements per user
    :param nz_user_itemindices: List of non-zero elements per item
    :return: User features matrix.
    """
    num_user = nnz_items_per_user.shape[0]
    num_feature = item_features.shape[0]
    lambda_I = lambda_user * sp.eye(num_feature)
    updated_user_features = np.zeros((num_feature, num_user))

    for user, items in nz_user_itemindices:
        # extract the columns corresponding to the prediction for given item
        M = item_features[:, items]

        # update column row of user features
        V = M @ train[items, user]
        A = M @ M.T + nnz_items_per_user[user] * lambda_I
        X = np.linalg.solve(A, V)
        updated_user_features[:, user] = np.copy(X.T)
    return updated_user_features


def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """
    Update item feature matrix.
    :param train: Matrix of ratings from train set
    :return: User features' matrix.
    :param user_features: Matrix of user features
    :param lambda_item: Float value of regularization parameter for item's matrix
    :param nnz_users_per_item: List of non-zero elements per item
    :param nz_item_userindices: List of non-zero elements per user
    :return: Item features matrix
    """
    num_item = nnz_users_per_item.shape[0]
    num_feature = user_features.shape[0]
    lambda_I = lambda_item * sp.eye(num_feature)
    updated_item_features = np.zeros((num_feature, num_item))

    for item, users in nz_item_userindices:
        # extract the columns corresponding to the prediction for given user
        M = user_features[:, users]
        V = M @ train[item, users].T
        A = M @ M.T + nnz_users_per_item[item] * lambda_I
        X = np.linalg.solve(A, V)
        updated_item_features[:, item] = np.copy(X.T)
    return updated_item_features


def ALS(train, test, lambda_user, lambda_item, num_features):
    """
    Alternating Least Squares (ALS) algorithm.
    :param train: Matrix of ratings from train set
    :param test: Matrix of ratings from test set
    :param lambda_user: Float value of regularization parameter for user's matrix
    :param lambda_item: Float value of regularization parameter for item's matrix
    :param num_features: Number of features
    :return: Item and users features matrices. RMSE on test set
    """
    # define parameters
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]

    # set seed
    np.random.seed(988)

    # init ALS
    print("INITIALIZE MATRICES...")
    user_features, item_features = init_mf(train, num_features)

    # get the number of non-zero ratings for each user and item
    nnz_items_per_user, nnz_users_per_item = train.getnnz(axis=0), train.getnnz(axis=1)

    # group the indices by row or column index
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)

    # run ALS
    print("LEARN MATRIX FACTORIZATION USING ALS...")
    while change > stop_criterion:
        # update user feature & item feature
        user_features = update_user_feature(
            train, item_features, lambda_user,
            nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(
            train, user_features, lambda_item,
            nnz_users_per_item, nz_item_userindices)

        error = compute_error(train, user_features, item_features, nz_train)
        print("RMSE on training set: {}.".format(error))
        error_list.append(error)
        change = np.fabs(error_list[-1] - error_list[-2])

    # evaluate the test error
    nnz_row, nnz_col = test.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    rmse = compute_error(test, user_features, item_features, nnz_test)
    print("test RMSE after running ALS: {v}.".format(v=rmse))
    return item_features, user_features, rmse


def global_mean(train):
    """
    Baseline method: use the global mean.
    :param train: Matrix of ratings from train set
    :return: Mean of all non-zero elements
    """
    # find the non zero ratings in the train
    nonzero_train = train[train.nonzero()]

    # calculate the global mean
    global_mean_train = nonzero_train.mean()

    return global_mean_train


def users_mean(train):
    """
    Computes the mean of the ratings per user
    :param train: Matrix of ratings from train set
    :return: Mean of all non-zero elements of each user
    """
    user_means = []
    for user in range(train.shape[1]):
        user_means.append(train[train[:, user].nonzero()[0], user].mean())
    return user_means


def items_mean(train):
    """
    Computes the mean of the ratings per item
    :param train: Matrix of ratings from train set
    :return: Mean of all non-zero elements of each item
    """
    item_means = []
    for item in range(train.shape[0]):
        item_means.append(train[item, train[item, :].nonzero()[1]].mean())
    return item_means


def compute_std(train):
    """
    Computes the std of the ratings per item
    :param train: Matrix of ratings from train set
    :return: Standard deviation of all non-zero elements of the train matrix
    """
    nonzero_train = train[train.nonzero()].toarray()
    return np.std(nonzero_train)


def standardize(train, mean):
    """
    Subtracts the global mean to all the non-zero elements of the matrix
    :param train: Matrix of ratings from train set
    :param mean: Global mean of all non-zero elements
    :return: Standardized train matrix
    """
    nz_row, nz_col = train.nonzero()
    for i in range(len(nz_row)):
        train[nz_row[i], nz_col[i]] -= mean
    return train


def div_std(train):
    """
    Divides by the standard deviation to all the non-zero elements of the matrix
    :param train: Matrix of ratings from train set
    :return: Standardized train matrix
    """
    std = compute_std(train)
    nz_row, nz_col = train.nonzero()
    for i in range(len(nz_row)):
        train[nz_row[i], nz_col[i]] /= std
    return train
