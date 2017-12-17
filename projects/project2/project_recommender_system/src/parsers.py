from datetime import datetime
import pandas as pd
import csv
import os
import pickle

from surprise import Reader, Dataset

DATA_DIR = "../data"
OUT_DIR = "../out"
TMP_DIR = "../tmp"


def load_data(filename):
    ds = load_pickle_data("ds")
    if ds is None:
        data = load_csv_data("{}/{}".format(DATA_DIR, filename))[1:]
        df = preprocess_data(data)
        reader = Reader(rating_scale=(1, 5))
        ds = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
        dump_pickle_data(ds, "ds")
    return ds


def load_csv_data(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def load_pickle_data(filename):
    """
    Loads data from .pckl file

    :param filename: Filename ({filename}.pckl) of the file located in the TMP_DIR directory
    :return: The object if the file existed, None otherwise
    """
    path = "{}/{}.pckl".format(TMP_DIR, filename)
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
    :param filename: Filename of the object (will be save as {TMP_DIR}/{filename}.pckl)
    :return: None
    """
    path = "{}/{}.pckl".format(TMP_DIR, filename)
    f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()


def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""

    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), int(rating)

    data = [deal_line(line) for line in data]

    ratings_dict = {
        'userID': [],
        'itemID': [],
        'rating': []
    }
    for row, col, rating in data:
        ratings_dict['userID'].append(row)
        ratings_dict['itemID'].append(col)
        ratings_dict['rating'].append(rating)

    return pd.DataFrame(ratings_dict)


def create_submission(algo):
    def deal_line(line):
        pos, _ = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return row, col

    data = load_csv_data("{}/sample_submission.csv".format(DATA_DIR))[1:]
    cells = [deal_line(line) for line in data]

    with open("{}/submission-{}.csv".format(OUT_DIR, datetime.now()), 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for c in cells:
            id = "r{}_c{}".format(c[0], c[1])
            pred = algo.predict(int(c[0]), int(c[1]))
            writer.writerow({'Id': id, 'Prediction': int(round(pred.est))})
