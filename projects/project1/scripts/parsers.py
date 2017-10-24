import numpy as np
import pickle
import os
import csv

DATA_DIR = "../data"


def load_data():
    print("PARSING TRAIN")
    ys_train, x_train, ids_train = load_pickle_data("ys_train"), load_pickle_data("x_train"), load_pickle_data(
        "ids_train")
    if ys_train is None or x_train is None or ids_train is None:
        ys_train, x_train, ids_train = load_csv_data("{}/train.csv".format(DATA_DIR))
        dump_pickle_data(ys_train, "ys_train")
        dump_pickle_data(x_train, "x_train")
        dump_pickle_data(ids_train, "ids_train")

    print("PARSING TEST")
    x_test, ids_test = load_pickle_data("x_test"), load_pickle_data("ids_test")
    if x_test is None or ids_test is None:
        _, x_test, ids_test = load_csv_data("{}/test.csv".format(DATA_DIR))
        dump_pickle_data(x_test, "x_test")
        dump_pickle_data(ids_test, "ids_test")

    return ys_train, x_train, ids_train, x_test, ids_test


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def load_pickle_data(filename):
    path = "../tmp/{}.pckl".format(filename)
    if os.path.exists(path):
        f = open(path, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj


def dump_pickle_data(obj, filename):
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
