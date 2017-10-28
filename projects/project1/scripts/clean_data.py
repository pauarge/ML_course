from filters import remove_bad_data, remove_good_data
from parsers import load_data
import numpy as np

def look_for_999(x):
    b_v = []
    for i in range(x.shape[1]):
        if np.min(x[:,i]) == -999:
            b_v.append(i)
    return b_v


def clean():
    ys_train, x_train, ids_train, x_test, ids_test = load_data()

    print("FILTERING DATA")
    # proves per least_squares sense rows amb -999
    x_train1, ys_train1 = remove_bad_data(x_train, ys_train)
    x_train2, ys_train2 = remove_good_data(x_train, ys_train)

    cm = np.corrcoef(x_train1, rowvar = False)
    bad_rows = look_for_999(x_train)
    #x_test, x_train1 = standardize(x_test, x_train1)
    #x_test2, x_train2 = standardize(x_test, x_train)


    x_train, ys_train = discard_outliers(x_train, ys_train, 1.95)
