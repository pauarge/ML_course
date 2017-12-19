from surprise import NMF, accuracy
from surprise import GridSearch, KNNBaseline
from surprise.prediction_algorithms.matrix_factorization import NMF, SVD
import pandas as pd
import argparse

from parsers import load_data



def main():
    print("LOADING DATAFRAME")
    data = load_data("data_train.csv")
    # data.split(5)
    te_error = []
    tr_error = []
    n_epochs = [150]
    n_facts = [70,80,90]
    for j in n_facts:
        algo = NMF(n_factors=j, n_epochs=150, reg_pu=0.1, reg_qi=0.1, biased=False, verbose=True)

        # grid_search = GridSearch(algo, param_grid, measures=['MAE', 'RMSE'], verbose=2)

        data.split(5)
        rmse_train = []
        rmse_test = []
        for i, (trainset_cv, testset_cv) in enumerate(data.folds()):
            print('fold number', i + 1)
            algo.train(trainset_cv)

            print('On testset,', end='  ')
            predictions = algo.test(testset_cv)
            rmse_test.append(accuracy.rmse(predictions, verbose=False))

            print('On trainset,', end=' ')
            predictions = algo.test(trainset_cv.build_testset())
            rmse_train.append(accuracy.rmse(predictions, verbose=False))
        print("test error for {} epochs: {}".format(j, sum(rmse_test) / len(rmse_test)))
        print("train error for {} epochs: {}".format(j, sum(rmse_train) / len(rmse_train)))
        te_error.append(sum(rmse_test) / len(rmse_test))
        tr_error.append(sum(rmse_train) / len(rmse_train))

    #cross_validation_epochs(n_epochs, tr_error, te_error, "RMSE_N_EPOCHS")

    # print("EVALUATING GRID SEARCH")
    # # Evaluate performances of our algorithm on the dataset.
    # grid_search.evaluate(data)
    #
    # print("BUILDING RESULTS")
    # results_df = pd.DataFrame.from_dict(grid_search.cv_results)
    # print(results_df)


if __name__ == '__main__':
    main()
