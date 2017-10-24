from datetime import datetime

from filters import discard_outliers, standarize
from implementations import least_squares, build_poly
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from helpers import dump_data, load_data

DATA_DIR = "../data"
OUT_DIR = "../out"


def main():
    print("PARSING TRAIN")
    ys_train, x_train, ids_train = load_data("ys_train"), load_data("x_train"), load_data("ids_train")
    if ys_train is None or x_train is None or ids_train is None:
        ys_train, x_train, ids_train = load_csv_data("{}/train.csv".format(DATA_DIR))
        dump_data(ys_train, "ys_train")
        dump_data(x_train, "x_train")
        dump_data(ids_train, "ids_train")

    print("PARSING TEST")
    x_test, ids_test = load_data("x_test"), load_data("ids_test")
    if x_test is None or ids_test is None:
        _, x_test, ids_test = load_csv_data("{}/test.csv".format(DATA_DIR))
        dump_data(x_test, "x_test")
        dump_data(ids_test, "ids_test")

    print("FILTERING DATA")
    x_test, x_train = standarize(x_test, x_train)
    x_train, ys_train = discard_outliers(x_train, ys_train, 2)

    print("BUILDING POLYNOMIALS")
    tx_train = build_poly(x_train, 3)
    tx_test = build_poly(x_test, 3)

    print("PREDICTING VALUES")
    w, loss = least_squares(ys_train, tx_train)
    y_pred = predict_labels(w, tx_test)

    print("EXPORTING CSV")
    create_csv_submission(ids_test, y_pred, "{}/submission-{}.csv".format(OUT_DIR, datetime.now()))


if __name__ == '__main__':
    main()
