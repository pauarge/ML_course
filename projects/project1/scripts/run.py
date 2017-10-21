from datetime import datetime

from implementations import least_squares, build_poly
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission


def main():
    ys_train, x_train, ids_train = load_csv_data("../data/train.csv")
    _, x_test, ids_test = load_csv_data("../data/test.csv")

    tx_train = build_poly(x_train, 1)
    tx_test = build_poly(x_test, 1)
    w, _ = least_squares(ys_train, tx_train)
    y_pred = predict_labels(w, tx_test)

    create_csv_submission(ids_test, y_pred, "../out/submission-{}.csv".format(datetime.now()))


if __name__ == '__main__':
    main()
