from surprise import NMF

from parsers import create_submission, load_data


def main():
    print("LOADING DATAFRAME")
    data = load_data("data_train.csv")

    print("TRAINING MODEL")
    trainset = data.build_full_trainset()
    algo = NMF(n_factors=25, n_epochs=350, verbose=True)
    algo.train(trainset)

    print("CREATING SUBMISSION")
    create_submission(algo)


if __name__ == '__main__':
    main()
