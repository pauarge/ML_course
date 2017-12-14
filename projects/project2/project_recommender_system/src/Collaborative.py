import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

from utils.methods import global_mean, standarize, div_std
from utils.parsers import load_data


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


def main():

    train, test, transformation_user, transformation_item = load_data(1)
    #M = np.array([[0, 2,0,0, 0], [0,0,0,4,0], [0,5,0,0,3],
    #             [4, 0,0,0,0], [0,0, 1, 0, 5], [3, 0,0,0,0],
    #              [0, 0, 1, 0, 2], [1, 0, 2, 0, 3], ])

    item_similarity = pairwise_distances(train, metric='cosine')
    #user_similarity = pairwise_distances(train.T, metric='cosine')

    similarity = (np.ones((10000,10000)) - item_similarity)

    gamma = 0.01
    num_epochs = 1  # number of full passes through the train set
    lambda_item = 0.001
    # set seed
    np.random.seed(988)


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
        err = 0
        i = 0
        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = similarity[d,:]
            user_info = train[:, n]
            contributors = item_info[user_info.nonzero()[0]].sum()
            user_info = user_info.toarray()
            err += (train[d, n] - (item_info.dot(user_info))/contributors)**2
            i +=1
            if i%100 ==0:
                print(i)
        print(np.sqrt(err/i))

            # calculate the gradient and update
            #similarity[:, d] += gamma * (err * user_info - lambda_item * item_info)



    #
    # item_prediction = predict(train, item_similarity, type='item')
    # user_prediction = predict(train, user_similarity, type='user')
    # print
    # 'User-based CF RMSE: ' + str(rmse(user_prediction, test))
    # print
    # 'Item-based CF RMSE: ' + str(rmse(item_prediction, test))

if __name__ == '__main__':
        main()