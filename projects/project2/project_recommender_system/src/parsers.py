from datetime import datetime
import pandas as pd
import csv

from surprise import Reader, Dataset

DATA_DIR = "../data"


def load_data(filename):
    data = read_txt("{}/{}".format(DATA_DIR, filename))[1:]
    df = preprocess_data(data)
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)


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

    # parse each line
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

    data = read_txt("{}/sample_submission.csv".format(DATA_DIR))[1:]
    cells = [deal_line(line) for line in data]

    with open("{}/submission-{}.csv".format(DATA_DIR, datetime.now()), 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for c in cells:
            id = "r{}_c{}".format(c[0], c[1])
            pred = algo.predict(int(c[0]), int(c[1]))
            writer.writerow({'Id': id, 'Prediction': int(round(pred.est))})
