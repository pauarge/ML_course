from scripts.proj1_helpers import load_csv_data, create_csv_submission, predict_labels
from scripts.implementations import least_squares


def main():
    ys_train, input_data_train, ids_train = load_csv_data("../data/train.csv")
    _, input_data_test, ids_test = load_csv_data("../data/test.csv")
    w, _ = least_squares(ys_train, input_data_train)
    y_pred = predict_labels(w, input_data_test)
    create_csv_submission(ids_test, y_pred, "out.csv")


if __name__ == '__main__':
    main()
