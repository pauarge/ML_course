import numpy as np
import scipy.sparse as sp
from itertools import groupby


def split_data(ratings, num_items_per_user, num_users_per_item, min_num_ratings, p_test=0.5):
    """
    split the ratings to training data and test data.

    :param ratings: Matrix of ratings
    :param num_items_per_user: Total number of items rated per user
    :param num_users_per_item: Total number of users who have rated an item
    :param min_num_ratings: Minimum number of ratings for an user or item to be included in the dataset
    :param p_test: Percentage of the data to set aside for the test set
    :return: List of valid_ratings, train and test datasets, transformation user and transformation item.
    """

    # set seed
    np.random.seed(1)

    valid_ratings, transformation_user, transformation_item = transformation(ratings, num_items_per_user,
                                                                             num_users_per_item, min_num_ratings)

    # init
    num_rows, num_cols = valid_ratings.shape
    train = sp.lil_matrix((num_rows, num_cols))
    test = sp.lil_matrix((num_rows, num_cols))

    print("the shape of original ratings. (# of row, # of col): {}".format(ratings.shape))
    print("the shape of valid ratings. (# of row, # of col): {}".format((num_rows, num_cols)))

    nz_items, nz_users = valid_ratings.nonzero()

    # split the data
    for user in set(nz_users):
        # randomly select a subset of ratings
        row, col = valid_ratings[:, user].nonzero()
        selects = np.random.choice(row, size=int(len(row) * p_test))
        residual = list(set(row) - set(selects))

        # add to train set
        train[residual, user] = valid_ratings[residual, user]

        # add to test set
        test[selects, user] = valid_ratings[selects, user]

    print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test, transformation_user, transformation_item


def split_data_2(elems, ratings, ratio, seed=1):
    """
    split the dataset based on the split ratio.
    :param elems: Numpy array with all the non-zero combinations of user-item from the original data
    :param ratings: Numpy array with all the non-zero ratings from the original data
    :param ratio: Float representing the ratio of data used for the test set
    :param seed:
    :return: Both elems and ratings split for train and test
    """
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(ratings)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    return elems[index_tr], ratings[index_tr], elems[index_te], ratings[index_te]


def transformation(ratings, num_items_per_user, num_users_per_item, min_num_ratings):
    """
    Generates transformation vector to correct the test matrix if users or items have been
    previously discarded from the data.
    :param ratings: Matrix of ratings
    :param num_items_per_user: Total number of items rated per user
    :param num_users_per_item: Total number of users who have rated an item
    :param min_num_ratings: Minimum number of ratings for an user or item to be included in the dataset
    :return: List of valid_ratings, transformation user and transformation item.
    """
    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][:, valid_users]

    # generate transformation vectors

    transformation_user = np.array(list(range(ratings.shape[1])))
    transformation_item = np.array(list(range(ratings.shape[0])))

    deleted_user = np.where(num_items_per_user < min_num_ratings)[0]
    deleted_item = np.where(num_users_per_item < min_num_ratings)[0]

    transformation_user[deleted_user] = -1
    transformation_item[deleted_item] = -1

    k = 0
    for i in range(len(transformation_user)):
        if transformation_user[i] == -1:
            k += 1
        else:
            transformation_user[i] -= k

    k_item = 0
    for i in range(len(transformation_item)):
        if transformation_item[i] == -1:
            k_item += 1
        else:
            transformation_item[i] -= k_item

    return valid_ratings, transformation_user, transformation_item


def plot_raw_data(ratings):
    """
    Plot the statistics result on a raw rating data
    :param ratings: Matrix of ratings
    :return: generates plot and returns two numpy arrays
    """
    """plot the statistics result on raw rating data."""
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    return num_items_per_user, num_users_per_item


def calculate_mse(real_label, prediction):
    """
    Calculate MSE.
    :param real_label: Real rating from the data
    :param prediction: Predicted rating
    :return: MSE
    """
    t = real_label - prediction
    return 1.0 * t.dot(t.T)


def build_index_groups(train):
    """
    Build groups for nnz rows and cols.
    :param train: Matrix of ratings from the train set
    :return: List of non-zero elements' indexes, list of non-zero rows and list of non-zero columns.
    """
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices


def group_by(data, index):
    """
    Group list of list by a specific index.
    :param data: List to group
    :param index: Indexes of the list
    :return: groups of lists
    """
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold
    N = #data points
    D = #number of variables in input data

    :param y: Vector of labels of size 1xN
    :param k_fold: Number of folds in which data is split
    :param seed: Number used to initialize a pseudorandom number generator.
    :return: k_fold vectors of size (1xN/k_fold) in which indices of input data are saved
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)
