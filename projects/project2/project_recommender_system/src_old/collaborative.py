import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt


def predict(ratings, similarity, pred_type='user'):
    if pred_type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
            [np.abs(similarity).sum(axis=1)]).T
    elif pred_type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    else:
        pred = None
    return pred


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


def collaborative(train):
    item_similarity = pairwise_distances(train, metric='cosine')

    # compute real similarity
    similarity = (np.ones((10000, 10000)) - item_similarity)

    gamma = 0.01

    # set seed
    np.random.seed(988)

    # find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    # shuffle the training rating indices
    np.random.shuffle(nz_train)

    # decrease step size
    gamma /= 1.2

    err = 0
    i = 0
    for d, n in nz_train:
        # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
        item_info = similarity[d, :]
        user_info = train[:, n]
        contributors = item_info[user_info.nonzero()[0]].sum()
        user_info = user_info.toarray()
        err += (train[d, n] - (item_info.dot(user_info)) / contributors) ** 2
        i += 1
        if i % 100 == 0:
            print("ITERATION: {}".format(i))
    rmse = np.sqrt(err / i)
    print(rmse)
    return similarity, train, rmse
