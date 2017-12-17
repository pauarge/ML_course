from surprise import NMF, Reader, GridSearch
from surprise import Dataset
import pandas as pd

from parsers import load_csv_data, create_submission


def cross_validation():
    df = load_csv_data("../data/data_train.csv")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    data.split(5)

    param_grid = {'n_factors': [15, 25], 'n_epochs': [50, 100]}
    grid_search = GridSearch(NMF, param_grid, measures=['RMSE'])

    # Evaluate performances of our algorithm on the dataset.
    grid_search.evaluate(data)

    results_df = pd.DataFrame.from_dict(grid_search.cv_results)
    print(results_df)


def main():
    print("LOADING DATASET")
    df = load_csv_data("../data/data_train.csv")

    print("CREATING DATAFRAME")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

    print("TRAINING MODEL")
    trainset = data.build_full_trainset()
    algo = NMF(n_factors=25, n_epochs=350, verbose=True)
    algo.train(trainset)

    print("CREATING SUBMISSION")
    create_submission(algo)


if __name__ == '__main__':
    main()
