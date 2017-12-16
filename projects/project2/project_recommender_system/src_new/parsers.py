from datetime import datetime
import pandas as pd
import csv

DATA_DIR = "../data"


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
