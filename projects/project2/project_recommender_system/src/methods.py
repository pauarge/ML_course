import numpy as np
import scipy as sp
from helpers import build_index_groups



def init_mf(train, num_features):
    """init the parameter for matrix factorization."""

    num_item, num_user = train.get_shape()

    # U, S, V  = np.linalg.svd(train)

    # user_features = U
    # item_features = S.dot(V.T)
    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)

    # start by item features.
    item_nnz = train.getnnz(axis=1)
    item_sum = np.array(train.sum(axis=1))
    # user_sum = train.sum(axis=0)
    # user_nnz = train.getnnz(axis=0)
    #
    print("GENERATING SPECIAL FEATURE")
    for ind in range(num_item):
        if item_nnz[ind] != 0:
            #print("item sum{}".format(item_sum[ind,0]))
            #print("item nnz{}".format(item_nnz[ind]))
            item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]
    # for ind in range(num_user):
    #     if user_nnz[ind] != 0:
    #         user_features[0, ind] = user_sum[0, ind] / user_nnz[ind]
    print(user_features.shape, item_features.shape)
    return user_features, item_features


def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    mse = 0
    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        mse += (data[row, col] - user_info.T.dot(item_info)) ** 2
    return np.sqrt(1.0 * mse / len(nz))


def compute_error_SVD(data, user_features, item_features, nz, mean, std):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    mse = 0
    pred = item_features.T.dot(user_features)
    i = 1
    for row, col in nz:
        #item_info = item_features[:,row]
        #user_info = user_features[:,col]
        #mse += (data[row, col] - item_info.dot(user_info.T)[0,0]) ** 2
        mse += (data[row,col]-(std*pred[row,col]+mean))**2
        if i%1000==0:
            print(mse/i)
        i += 1
    return np.sqrt(1.0 * mse / len(nz))


def matrix_factorization_SGD(train, test, lambda_user, lambda_item, num_features):
    """matrix factorization by SGD."""
    # define parameters
    gamma = 0.01
    #num_features = 2  # K in the lecture notes
    # lambda_user = 0.1
    # lambda_item = 0.01
    num_epochs = 20 # number of full passes through the train set

    # set seed
    np.random.seed(988)

    mean = global_mean(train)
    train = standarize(train,mean)
    std = compute_std(train)
    train = div_std(train)


    # init matrix
    user_features, item_features = init_mf(train, num_features)

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
            err = train[d, n] - user_info.T.dot(item_info)

            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)

        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        print("iter: {}".format(it))

        # errors.append(rmse)

    # evaluate the test error
    rmse = compute_error_SVD(test, user_features, item_features, nz_test, mean, std)
    print("RMSE on test data: {}.".format(rmse))
    return item_features, user_features, rmse


# ALTERNATING LEAST SQUARES

def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""
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
    """update item feature matrix."""
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
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    # num_features = 20  # K in the lecture notes
    # lambda_user = 0.1
    # lambda_item = 0.7
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]

    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_mf(train, num_features)

    # get the number of non-zero ratings for each user and item
    nnz_items_per_user, nnz_users_per_item = train.getnnz(axis=0), train.getnnz(axis=1)

    # group the indices by row or column index
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)

    # run ALS
    print("start the ALS algorithm...")
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
    """baseline method: use the global mean."""
    # find the non zero ratings in the train
    nonzero_train = train[train.nonzero()]

    # calculate the global mean
    global_mean_train = nonzero_train.mean()

    return global_mean_train


def user_mean(train, user):
    """compute user mean"""
    return train[:, user].mean()


def item_mean(train, item):
    """compute item mean"""
    return train[item, :].mean()

def compute_std(train):

    nonzero_train = train[train.nonzero()].toarray()
    return np.std(nonzero_train)


def standarize(train,mean):

    nz_row, nz_col = train.nonzero()
    for i in range(len(nz_row)):
        train[nz_row[i], nz_col[i]] -= mean
    return train


def div_std(train):
    std = compute_std(train)
    nz_row, nz_col = train.nonzero()
    for i in range(len(nz_row)):
        train[nz_row[i], nz_col[i]] /= std
    return train