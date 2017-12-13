import numpy as np

from methods import global_mean, standarize, compute_std, div_std, users_mean, items_mean, \
    matrix_factorization_SGD, matrix_factorization_sgd_std, compute_error_bias
from parsers import load_data, create_submission


def run():
    # define parameters for simulation
    lambda_user = 0.1
    lambda_item = 0.01
    num_features = 2
    min_num_data = 1

    # load data train csv
    print("LOADING DATA...")
    train, test, transformation_user, transformation_item = load_data(min_num_data)

    # define data parameters
    print("STARTING MATRIX FACTORIZATION SGD")
    mean = global_mean(train)
    print("CALCULATE BIAS")
    users_bias = users_mean(train) - mean * np.ones((1, train.shape[1]))  # sense estandaritzar
    print("USER BIAS")
    items_bias = items_mean(train) - mean * np.ones((1, train.shape[0]))  # sense estandaritzar
    print("ITEM BIAS")

    users_bias = users_bias.flatten()
    items_bias = items_bias.flatten()

    # standarization
    train = standarize(train, mean)
    std = compute_std(train)
    train = div_std(train)

    u_bias = users_bias/std
    i_bias = items_bias/std

    item_features, user_features = matrix_factorization_sgd_std(train, lambda_user, lambda_item, num_features, u_bias, i_bias)

    # evaluate the test error
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    rmse, rmse_r, rmse_g = compute_error_bias(test, user_features, item_features, nz_test, mean, std, users_bias, items_bias)
    print("RMSE on test data: {}.".format(rmse))
    print("RMSE_ROUND on test data: {}.".format(rmse_r))
    print("RMSE_GROUND on test data: {}.".format(rmse_g))

    create_submission(item_features, user_features, train, transformation_user, transformation_item, mean, std,
                      users_bias, items_bias)
    return rmse


if __name__ == '__main__':
    run()
