import numpy as np
import scipy.sparse as sp
from itertools import groupby


def split_data(ratings, num_items_per_user, num_users_per_item, min_num_ratings, p_test=0.2):
    """
    split the ratings to training data and test data.
    Args:
        min_num_ratings:
            all users and items we keep must have at least min_num_ratings per user and per item.
    """
    # set seed
    np.random.seed(1)

    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][:, valid_users]

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
    return valid_ratings, train, test


def plot_raw_data(ratings):
    """plot the statistics result on raw rating data."""
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    return num_items_per_user, num_users_per_item


def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)


def build_index_groups(train):
    """build groups for nnz rows and cols."""
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
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


