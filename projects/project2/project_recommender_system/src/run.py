from surprise import NMF
from datetime import datetime
import argparse

from surprise.dump import dump

from parsers import create_submission, load_data, TMP_DIR


def main():
    print("PARSING CMD ARGS")
    parser = argparse.ArgumentParser(description='Cross validation for movie ratings.')
    parser.add_argument('--epochs', '-e', default=500, type=int, help="Number of epochs to test.")
    parser.add_argument('--verbose', '-v', default=True, type=bool, help="Set verbosity of ")
    parser.add_argument('--biased', '-b', default=False, type=bool, help="Run the method biased")
    parser.add_argument('--factors', '-f', default=25, type=int, help="Number of factors")
    args = parser.parse_args()

    print("LOADING DATAFRAME")
    data = load_data("data_train.csv")

    print("TRAINING MODEL")
    trainset = data.build_full_trainset()
    algo = NMF(n_factors=args.factors, n_epochs=args.epochs, biased=args.biased, verbose=args.verbose)
    algo.train(trainset)

    print("CREATING SUBMISSION")
    create_submission(algo)

    print("SAVING MODEL")
    dump("{}/suprise-{}.pckl".format(TMP_DIR, datetime.now()), algo=algo)


if __name__ == '__main__':
    main()
