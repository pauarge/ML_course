from datetime import datetime

from filters import discard_outliers, standardize
from helpers import predict_labels
from implementations import least_squares, build_poly
from parsers import load_data, create_csv_submission

OUT_DIR = "../out"


def main():
    ys_train, x_train, ids_train, x_test, ids_test = load_data()

    print("FILTERING DATA")
    x_test, x_train = standardize(x_test, x_train)
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
