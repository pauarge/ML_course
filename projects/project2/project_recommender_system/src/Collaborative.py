import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

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
    train, test, transformation_user, transformation_item = load_data(min_num_data)
    user_similarity = pairwise_distances(train, metric='cosine')  ######ALERTA POTSER ES AL REVES EL TRASPOSAT
    item_similarity = pairwise_distances(train.T, metric='cosine')

    item_prediction = predict(train_data_matrix, item_similarity, type='item')
    user_prediction = predict(train_data_matrix, user_similarity, type='user')
    print
    'User-based CF RMSE: ' + str(rmse(user_prediction, test))
    print
    'Item-based CF RMSE: ' + str(rmse(item_prediction, test))

if __name__ == '__main__':
        run()