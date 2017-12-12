import scipy.sparse as sp
import numpy as np
import pickle
import os
import csv

from helpers import plot_raw_data, split_data, split_data_2
from methods import user_mean, global_mean, item_mean

DATA_DIR = "../data"


def load_data(min_num_ratings):
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
    t_u = load_pickle_data("t_u")
    t_i = load_pickle_data("t_i")
    if train is None or test is None:
        num_items_per_user, num_users_per_item = plot_raw_data(matrix_train)
        valid_data, train, test, t_u, t_i = split_data(matrix_train, num_items_per_user, num_users_per_item,
                                                       min_num_ratings)
        dump_pickle_data(train, "train")
        dump_pickle_data(test, "test")
        dump_pickle_data(t_u, "t_u")
        dump_pickle_data(t_i, "t_i")

    return train, test, t_u, t_i


def load_data_2():
    """
    Loads datasets into the program. If they exist, it loads the cached .pckl files, it loads the .csv otherwise

    :return: Numpy arrays containing train ys, x, ids and test x, ids
    """
    print("PARSING TRAIN")
    elems = load_pickle_data("elems")
    ratings = load_pickle_data("ratings")
    if elems is None or ratings is None:
        elems, ratings = load_csv_data_2("{}/data_train.csv".format(DATA_DIR))
        dump_pickle_data(elems, "elems")
        dump_pickle_data(ratings, "ratings")

    return split_data_2(elems, ratings, 0.2)


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


def load_csv_data_2(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data_2(data)


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

    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def preprocess_data_2(data):
    """preprocessing the text data, conversion to numerical array format."""

    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    # parse each line
    data = [deal_line(line) for line in data]

    elems = np.empty([len(data), 2], dtype=np.float32)
    ratings = np.empty([len(data), 1], dtype=np.float32)
    for i, data in enumerate(data):
        elems[i][0] = data[0]
        elems[i][1] = data[1]
        ratings[i] = data[2]
    return elems, ratings


def create_submission(w, z, train, trans_user, trans_item, mean, std):
    print("CREATING SUBMISSION")
    def deal_line(line):
        pos, _ = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row) - 1, int(col) - 1

    data = read_txt("{}/sample_submission.csv".format(DATA_DIR))[1:]
    cells = [deal_line(line) for line in data]

    x = std*np.transpose(w).dot(z)+mean

    # ids = ["r{}_c{}".format(c[0] + 1, c[1] + 1) for c in cells]
    # preds = [round(x[c[0], c[1]]) for c in cells]

    g_mean = global_mean(train)
    ids = []
    preds = []
    for c in cells:
        ids.append("r{}_c{}".format(c[0] + 1, c[1] + 1))
        if trans_item[c[0]] == -1 and trans_user[c[1]] == -1:
            preds.append(g_mean)
        elif trans_item[c[0]] == -1 and trans_user[c[1]] != -1:
            preds.append(user_mean(train, c[1]))
        elif trans_item[c[0]] != -1 and trans_user[c[1]] == -1:
            preds.append(item_mean(train, c[0]))
        else:
            preds.append(round(x[trans_item[c[0]], trans_user[c[1]]]))

    with open("{}/submission.csv".format(DATA_DIR), 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, preds):
            writer.writerow({'Id': r1, 'Prediction': int(r2)})


