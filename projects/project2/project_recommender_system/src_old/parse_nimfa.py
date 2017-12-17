from datetime import datetime
import numpy as np
import scipy.sparse as sp
import sys
import csv


def load_csv_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""

    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), int(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_row, max_col))

    ratings = sp.lil_matrix((max_row, max_col), dtype=np.int)
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def split_data(data, ratio=0.1, seed=1):
    np.random.seed(seed)

    num_rows, num_cols = data.shape
    train = sp.lil_matrix((num_rows, num_cols), dtype=np.int)
    test = sp.lil_matrix((num_rows, num_cols), dtype=np.int)

    nz_items, nz_users = data.nonzero()

    for user in set(nz_users):
        # randomly select a subset of ratings
        row, col = data[:, user].nonzero()
        selects = np.random.choice(row, size=int(len(row) * ratio))
        residual = list(set(row) - set(selects))

        # add to train set
        train[residual, user] = data[residual, user]

        # add to test set
        test[selects, user] = data[selects, user]

    return train, test


def create_submission(W, H):
    print("CREATING SUBMISSION")

    def deal_line(line):
        pos, _ = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row) - 1, int(col) - 1

    data = read_txt("data/sample_submission.csv")[1:]
    cells = [deal_line(line) for line in data]

    ids = []
    preds = []
    for c in cells:
        sc = max(min((W[c[0] - 1, :] * H[:, c[1] - 1])[0, 0], 5), 1)
        ids.append("r{}_c{}".format(c[0] + 1, c[1] + 1))
        preds.append(sc)

    with open("data/submission-{}.csv".format(datetime.now()), 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, preds):
            writer.writerow({'Id': r1, 'Prediction': int(r2)})


def main(argv):
    data = load_csv_data(argv[0])
    train, test = split_data(data)

    f = open("{}.train.txt".format(argv[0]), 'w')
    g = open("{}.test.txt".format(argv[0]), 'w')

    nz_train = train.nonzero()
    nz_test = test.nonzero()

    for i in range(len(nz_train[0])):
        f.write("{}\t{}\t{}\n".format(nz_train[0][i], nz_train[1][i], train[nz_train[0][i], nz_train[1][i]]))

    for i in range(len(nz_test[0])):
        g.write("{}\t{}\t{}\n".format(nz_test[0][i], nz_test[1][i], test[nz_test[0][i], nz_test[1][i]]))


if __name__ == '__main__':
    main(sys.argv[1:])
