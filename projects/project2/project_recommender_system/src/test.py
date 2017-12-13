import numpy as np

from utils.helpers import build_k_indices, split_data_3
from utils.parsers import load_data_3

from run_bias import run as run_bias


def main():
    lambda_user = 0
    lambda_item = 0
    num_features = 2
    min_num_data = 150
    seed = 3
    k_fold = 4

    ratings = load_data_3().toarray()
    indexes = split_data_3(ratings, k_fold)

    tr, te = 0, 0

    for j in range(k_fold):
        test = ratings[indexes[j]]
        new_indices = indexes
        del new_indices[j]
        flat_list = [item for sublist in new_indices for item in sublist]
        train = ratings[flat_list]

        run_bias(train, test, lambda_user, lambda_item, num_features)


if __name__ == '__main__':
    main()
