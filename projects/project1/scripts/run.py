from datetime import datetime

from clean_data import standardize, discard_outliers, remove_bad_data
from helpers import predict_labels, build_poly
from implementations import least_squares
from parsers import load_data, create_csv_submission

OUT_DIR = "../out"
OUTLIERS_THRESHOLD = 9.2
DEGREE = 13


def main():
    ys_train, x_train, ids_train, x_test, ids_test = load_data()

    print("FILTERING DATA")
    x_test, x_train = standardize(x_test, x_train)
    x_train, ys_train = discard_outliers(x_train, ys_train, OUTLIERS_THRESHOLD)

    print("BUILDING POLYNOMIALS")
    tx_train = build_poly(x_train, DEGREE)
    tx_test = build_poly(x_test, DEGREE)

    print("LEARNING MODEL BY LEAST SQUARES")
    w, mse = least_squares(ys_train, tx_train)

    print("PREDICTING VALUES")
    y_pred = predict_labels(w, tx_test)

    print("EXPORTING CSV")
    create_csv_submission(ids_test, y_pred, "{}/submission-{}.csv".format(OUT_DIR, datetime.now()))


if __name__ == '__main__':
    main()
