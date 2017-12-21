from surprise import NMF
from datetime import datetime
import argparse

from surprise.dump import dump, load

from parsers import create_submission, load_data, TMP_DIR


def parse_args():
    """
    Sets up a parser for CLI options.

    :return: arguments list
    """
    parser = argparse.ArgumentParser(description='Main program for getting rating predictions of the given dataset.')
    parser.add_argument('-b', '--biased', default=False, type=bool, help="Run bias on the method.")
    parser.add_argument('-e', '--epochs', default=500, type=int, help="Number of epochs to run.")
    parser.add_argument('-f', '--factors', default=60, type=int, help="Number of factors of the method.")
    parser.add_argument('-pu', '--reg_pu', default=0.1, type=float, help="Regularization factor for users.")
    parser.add_argument('-qi', '--reg_qi', default=0.1, type=float, help="Regularization factor for items.")
    parser.add_argument('-v', '--verbose', default=True, type=bool, help="Enable verbosity of training.")
    parser.add_argument('-m', '--model', type=str, help="Load previously calculated model")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model:
        _, algo = load("{}/{}".format(TMP_DIR, args.model))
        if algo is None:
            print("Could not load given dump")
            return
    else:
        print("LOADING DATAFRAME")
        data = load_data("data_train.csv", min_ratings_user=1)

        print("TRAINING MODEL")
        trainset = data.build_full_trainset()
        algo = NMF(n_factors=args.factors, n_epochs=args.epochs, biased=args.biased, verbose=args.verbose,
                   reg_pu=args.reg_pu, reg_qi=args.reg_qi)
        algo.train(trainset)

        print("SAVING MODEL")
        dump("{}/model-{}.pckl".format(TMP_DIR, datetime.now()), algo=algo)

    print("CREATING SUBMISSION")
    create_submission(algo)


if __name__ == '__main__':
    main()
