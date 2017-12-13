import numpy as np

from utils.methods import global_mean, standarize, compute_std, div_std, users_mean, items_mean, \
    matrix_factorization_sgd_std, decomposition_error
from utils.parsers import load_data, create_submission


def run(train, test, lambda_user, lambda_item, num_features):
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

    u_bias = users_bias / std
    i_bias = items_bias / std

    item_features, user_features = matrix_factorization_sgd_std(train, lambda_user, lambda_item, num_features, u_bias,
                                                                i_bias)

    # evaluate the test error
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    mse1, mse2, mse3, mse4, percentage, total \
        = decomposition_error(train, test, user_features, item_features, nz_test, mean, std, users_bias, items_bias, 50)

    print("RMSE on normal test data: {}.".format(mse1))
    print("RMSE on bad user data: {}.".format(mse2))
    print("RMSE on bad item data: {}.".format(mse3))
    print("RMSE on isolated data: {}.".format(mse4))
    print("TOTAL RMSE {}".format(total))
    print("PERCENTAGE: {}".format(percentage))

    return item_features, user_features, mean, std, users_bias, items_bias, total


def main():
    # define parameters for simulation
    lambda_user = 0
    lambda_item = 0
    num_features = 2
    min_num_data = 150

    # load data train csv
    print("LOADING DATA...")
    train, test, transformation_user, transformation_item = load_data(min_num_data)

    item_features, user_features, mean, std, users_bias, items_bias, _ = run(train, test, lambda_user, lambda_item,
                                                                             num_features)

    create_submission(item_features, user_features, train, transformation_user, transformation_item, mean, std,
                      users_bias, items_bias)


if __name__ == '__main__':
    main()
