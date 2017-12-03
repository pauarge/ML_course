import scipy.sparse as sp
import numpy as np
import pickle
import os
import csv

from helpers import plot_raw_data, split_data

DATA_DIR = "../data"
MIN_NUM_RATINGS = 1


def load_data():
    """
    Loads datasets into the program. If they exist, it loads the cached .pckl files, it loads the .csv otherwise

    :return: Numpy arrays containing train ys, x, ids and test x, ids
    """
    print("PARSING TRAIN")
    matrix_train = load_pickle_data("matrix_train")
    if matrix_train is None:
        matrix_train = load_csv_data("{}/data_train.csv".format(DATA_DIR))
        dump_pickle_data(matrix_train, "matrix_train")

    train = load_pickle_data("train")
    test = load_pickle_data("test")
    if train is None or test is None:
        num_items_per_user, num_users_per_item = plot_raw_data(matrix_train)
        valid_data, train, test = split_data(matrix_train, num_items_per_user, num_users_per_item, MIN_NUM_RATINGS)
        dump_pickle_data(train, "train")
        dump_pickle_data(test, "test")
    return train, test


def load_pickle_data(filename):
    """
    Loads data from .pckl file

    :param filename: Filename ({filename}.pckl) of the file located in the ../tmp directory
    :return: The object if the file existed, None otherwise
    """
    path = "../tmp/{}.pckl".format(filename)
    if os.path.exists(path):
        print("LOADING PCKL FILE FROM {}".format(path))
        f = open(path, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj


def dump_pickle_data(obj, filename):
    """
    Dumps the given object into a .pckl file

    :param obj: Valid Python object to dump
    :param filename: Filename of the object (will be save as ../tmp/{filename}.pckl)
    :return: None
    """
    path = "../tmp/{}.pckl".format(filename)
    f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def load_csv_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""

    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def create_submission(w, z):
    def deal_line(line):
        pos, _ = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row) - 1, int(col) - 1

    data = read_txt("{}/sample_submission.csv".format(DATA_DIR))[1:]
    cells = [deal_line(line) for line in data]

    x = np.transpose(w).dot(z)

    ids = ["r{}_c{}".format(c[0] + 1, c[1] + 1) for c in cells]
    preds = [round(x[c[0], c[1]]) for c in cells]

    with open("{}/submission.csv".format(DATA_DIR), 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, preds):
            writer.writerow({'Id': r1, 'Prediction': int(r2)})
