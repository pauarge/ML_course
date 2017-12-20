import argparse
import numpy as np

from collaborative import collaborative
from utils.SVD import SVD_SGD
from utils.methods import matrix_factorization_SGD, ALS, global_mean, standardize, compute_std, div_std, \
    users_mean, items_mean, matrix_factorization_sgd_std, matrix_factorization_sk, decomposition_error
from utils.parsers import load_data, create_submission


def parse_args():
    """
    Sets up a parser for CLI options.

    :return: arguments list
    """
    parser = argparse.ArgumentParser(
        description='Secondary methods for getting rating predictions of the given dataset.')

    parser.add_argument('-m', '--method', default='NMF_SGD', type=str, help="Chosen method among 'NMF_SGD', 'NMF_ALS', "
                                                                            "'NMF_SK', 'BIAS_NMF_SGD', 'SVD', 'COL'.")
    parser.add_argument('-f', '--features', default=25, type=int, help="Number of features of the method.")
    parser.add_argument('-l_u', '--lambda_user', default=0.001, type=float,
                        help="Regularization parameter for user matrix.")
    parser.add_argument('-l_i', '--lambda_item', default=0.01, type=float,
                        help="Regularization parameter for item matrix.")
    parser.add_argument('-d', '--min_data', default=1, type=int, help="Minimum number of ratings for a user or film "
                                                                      "to be included in the data.")
    parser.add_argument('-s', '--submission', default=False, type=bool, help="Create submission csv.")
    parser.add_argument('-e', '--error', default=False, type=bool, help="Generate error decomposition.")

    return parser.parse_args()


def main():
    args = parse_args()

    # get global parameters
    num_features = args.factors
    min_num_data = args.min_data

    # default arguments for non-standardized data
    mean = 0
    std = 1
    users_bias = 0
    items_bias = 0

    print("LOADING DATA...")
    train, test, transformation_user, transformation_item = load_data(min_num_data)
    item_features = None
    user_features = None

    if args.method == 'NMF_SGD':
        item_features, user_features, rmse_te = matrix_factorization_SGD(train, test, args.lambda_user,
                                                                         args.lambda_item, num_features)
    elif args.method == 'NMF_ALS':
        item_features, user_features, rmse_te = ALS(train, test, args.lambda_user, args.lambda_item, num_features)

    elif args.method == 'NMF_SK':
        item_features, user_features, rmse_te = matrix_factorization_sk(train, test, num_features, args.lambda_user)

    elif args.method == 'BIAS_NMF_SGD':

        mean = global_mean(train)

        print("CALCULATING BIAS...")
        users_bias = users_mean(train) - mean * np.ones((1, train.shape[1]))
        items_bias = items_mean(train) - mean * np.ones((1, train.shape[0]))

        users_bias = users_bias.flatten()
        items_bias = items_bias.flatten()

        # standardization
        train = standardize(train, mean)
        std = compute_std(train)
        train = div_std(train)

        u_bias = users_bias / std
        i_bias = items_bias / std

        item_features, user_features, rmse_te = \
            matrix_factorization_sgd_std(train, test, args.lambda_user, args.lambda_item, num_features, u_bias, i_bias)

    elif args.method == 'SVD':
        item_features, user_features, rmse = SVD_SGD(train, test, args.lambda_user, args.lambda_item, num_features)

    elif args.method == 'COL':
        item_features, user_features, rmse = collaborative(train)

    if args.submission:
        if item_features is None or user_features is None:
            print("PLEASE SPECIFY METHOD")
        else:
            create_submission(item_features, user_features, train, transformation_user, transformation_item, mean, std)

    if args.error:
        nz_row, nz_col = test.nonzero()
        nz_test = list(zip(nz_row, nz_col))

        mse1, mse2, mse3, mse4, percentage, total \
            = decomposition_error(train, test, user_features, item_features, nz_test, mean, std, users_bias, items_bias,
                                  min_num_data)

        print("RMSE on normal test data: {}.".format(mse1))
        print("RMSE on bad user data: {}.".format(mse2))
        print("RMSE on bad item data: {}.".format(mse3))
        print("RMSE on isolated data: {}.".format(mse4))
        print("TOTAL RMSE {}".format(total))
        print("PERCENTAGE: {}".format(percentage))


if __name__ == '__main__':
    main()
