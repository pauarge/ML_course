from surprise import GridSearch, KNNBaseline
from surprise.prediction_algorithms.matrix_factorization import NMF, SVD
import pandas as pd
import argparse

from parsers import load_data


def parse_args():
    """
    Sets up a parser for CLI options.

    :return:
    """
    parser = argparse.ArgumentParser(description='Cross validation for movie ratings.')
    parser.add_argument('algorithm', type=str, help="Algorithm to use. Options: SVD, NMF, KNNB")
    parser.add_argument('--epochs', '-e', default=100, type=int, help="Number of epochs to test.")
    return parser.parse_args(), parser


def main():
    args, parser = parse_args()

    print("LOADING DATAFRAME")
    data = load_data("data_train.csv")
    data.split(5)

    print("DEFINING PARAMETERS")
    if args.algorithm == "NMF":
        algo = NMF
        param_grid = {'n_factors': [10, 15, 20, 25, 30], 'n_epochs': [args.epochs], 'biased': [True, False]}
    elif args.algorithm == "SVD":
        algo = SVD
        param_grid = {'n_factors': [50, 75, 100, 125, 150], 'n_epochs': [args.epochs], 'biased': [True, False]}
    elif args.algorithm == "KNNB":
        algo = KNNBaseline
        param_grid = {'k': [20, 30, 40, 50, 60], 'min_k': [0, 1, 2, 3]}
    else:
        parser.print_usage()
        return

    grid_search = GridSearch(algo, param_grid, measures=['MAE', 'RMSE'], verbose=2)

    print("EVALUATING PERFORMANCE OF THE ALGORITHM")
    grid_search.evaluate(data)

    print("BUILDING RESULTS")
    results_df = pd.DataFrame.from_dict(grid_search.cv_results)
    print(results_df)


if __name__ == '__main__':
    main()
