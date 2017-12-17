from surprise import NMF
from datetime import datetime

from surprise.dump import dump

from parsers import create_submission, load_data, TMP_DIR


def main():
    print("LOADING DATAFRAME")
    data = load_data("data_train.csv")

    print("TRAINING MODEL")
    trainset = data.build_full_trainset()
    algo = NMF(n_factors=15, n_epochs=250, biased=True, verbose=True)
    algo.train(trainset)

    print("CREATING SUBMISSION")
    create_submission(algo)

    print("SAVING MODEL")
    dump("{}/suprise-{}.pckl".format(TMP_DIR, datetime.now()), algo=algo)


if __name__ == '__main__':
    main()
