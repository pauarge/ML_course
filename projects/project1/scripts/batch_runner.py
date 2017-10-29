from tornado import concurrent

from clean_data import standardize, discard_outliers
from helpers import build_poly, predict_labels
from implementations import least_squares, reg_logistic_regression
from parsers import load_data, create_csv_submission
from keggle import main as keggle
import glob

OUT_DIR = "../out"
MAX_ITERS = 1000000


def process_file_lr(x_train, x_test, ys_train, ids_test, o, d, l):
    print("CALCULATING LR {} {} {}".format(o, d, l))
    tx_train = build_poly(x_train, d)
    tx_test = build_poly(x_test, d)
    w, _ = least_squares(ys_train, tx_train)
    w, _ = reg_logistic_regression(ys_train, tx_train, l, w, MAX_ITERS, 0.1)
    y_pred = predict_labels(w, tx_test)
    create_csv_submission(ids_test, y_pred, "{}/batch-lr-{}-{}-{}.csv".format(OUT_DIR, o, d, l))


def process_file_le(x_train, x_test, ys_train, ids_test, o, d):
    print("CALCULATING LE {} {}".format(o, d))
    tx_train = build_poly(x_train, d)
    tx_test = build_poly(x_test, d)
    w, _ = least_squares(ys_train, tx_train)
    y_pred = predict_labels(w, tx_test)
    create_csv_submission(ids_test, y_pred, "{}/batch-le-{}-{}.csv".format(OUT_DIR, o, d))


def main():
    outliers = range(2, 13)
    degrees = range(1, 8)
    lambdas = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0]
    ys_train, x_train, ids_train, x_test, ids_test = load_data()

    x_test, x_train = standardize(x_test, x_train)

    executor = concurrent.futures.ThreadPoolExecutor(8)
    futures = []

    for o in outliers:
        x_train_in, ys_train_in = discard_outliers(x_train, ys_train, o)
        for d in degrees:
            for l in lambdas:
                futures.append(executor.submit(process_file_lr, x_train_in, x_test, ys_train_in, ids_test, o, d, l))

    concurrent.futures.wait(futures)

    futures = []

    for filename in glob.glob('../out/*.csv'):
        futures.append(executor.submit(keggle, [filename], True, True))

    concurrent.futures.wait(futures)


if __name__ == '__main__':
    main()
