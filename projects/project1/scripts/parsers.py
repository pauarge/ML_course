import numpy as np
import pickle
import os
import csv

DATA_DIR = "../data"


def load_data():
    """
    Loads datasets into the program. If they exist, it loads the cached .pckl files, it loads the .csv otherwise

    :return: Numpy arrays containing train ys, x, ids and test x, ids
    """
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


def load_csv_data(data_path):
    """
    Loads data and returns y (class labels), tX (features) and ids (event ids)

    :param data_path: Data path of the .csv file
    :return: Ys, input_data and ids loaded from those files
    """
    print("LOADING CSV FILE FROM {}".format(data_path))
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=[1])
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    return yb, input_data, ids


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
