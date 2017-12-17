import pandas as pd
from surprise import GridSearch
from surprise.prediction_algorithms.matrix_factorization import NMF

from parsers import load_data


def main():
    data = load_data("data_train.csv")
    data.split(5)

    algo = NMF
    param_grid = {'n_factors': [10, 15, 20, 25, 30], 'n_epochs': [100], 'biased': [True, False]}

    grid_search = GridSearch(algo, param_grid, measures=['MAE', 'RMSE'], verbose=2)

    # Evaluate performances of our algorithm on the dataset.
    grid_search.evaluate(data)

    results_df = pd.DataFrame.from_dict(grid_search.cv_results)
    print(results_df)


if __name__ == '__main__':
    main()
