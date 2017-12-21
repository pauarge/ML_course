from surprise import accuracy
from surprise.prediction_algorithms.matrix_factorization import NMF

from parsers import load_data


def main():
    print("LOADING DATAFRAME")
    data = load_data("data_train.csv")
    # data.split(5)
    te_error = []
    tr_error = []
    n_epochs = [100, 250]
    n_facts = [65, 80, 95, 110]
    reg = [0.1]
    for n in n_epochs:
        for k in n_facts:
            for l in reg:
                algo = NMF(n_factors=k, n_epochs=n, reg_pu=l, reg_qi=l, biased=False, verbose=True)

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
                print("test error for {} epochs {} factors: {}".format(n, k, sum(rmse_test) / len(rmse_test)))
                print("train error for {} epochs {} factors : {}".format(n, k, sum(rmse_train) / len(rmse_train)))
                te_error.append(sum(rmse_test) / len(rmse_test))
                tr_error.append(sum(rmse_train) / len(rmse_train))


if __name__ == '__main__':
    main()
