from SVD import SGD
import numpy as np
from helpers import calculate_mse
from methods import matrix_factorization_SGD, ALS
from parsers import load_data, create_submission


def run(lambda_user=0.1, lambda_item=0.01, num_features=30, min_num_data=1, p_test=0.2):
    print("LOADING DATA...")
    train, test, transformation_user, transformation_item = load_data(min_num_data)

    # urm = np.zeros(train.shape)

    num_features = 2
    lambda_user = 0.01
    lambda_item = 0.001
    item_features, user_features, rmse = SGD(train, test, lambda_user, lambda_item, num_features)

    print("ERROR TEST by SVD: {}".format(rmse))


if __name__ == '__main__':
    run()
