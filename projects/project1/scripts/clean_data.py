from parsers import load_data
import numpy as np


def look_for_999(x):
    b_v = []
    for i in range(x.shape[1]):
        if np.min(x[:, i]) == -999:
            b_v.append(i)
    return b_v


def clean():
    ys_train, x_train, ids_train, x_test, ids_test = load_data()

    print("FILTERING DATA")
    # proves per least_squares sense rows amb -999
    x_train1, ys_train1 = remove_bad_data(x_train, ys_train)
    x_train2, ys_train2 = remove_good_data(x_train, ys_train)

    cm = np.corrcoef(x_train1, rowvar=False)
    bad_rows = look_for_999(x_train)
    # x_test, x_train1 = standardize(x_test, x_train1)
    # x_test2, x_train2 = standardize(x_test, x_train)


    x_train, ys_train = discard_outliers(x_train, ys_train, 1.95)


def standardize(x_test, x_train):
    for i in range(x_test.shape[1]):
        x_test[:, i], x_train[:, i] = standardize_col(x_test[:, i], x_train[:, i])
    return x_test, x_train


def standardize_col(x1, x2):
    index_x1 = np.where(x1 == -999)
    index_x2 = np.where(x2 == -999)

    x1_clean = np.delete(x1, index_x1)
    x2_clean = np.delete(x2, index_x2)
    x_clean = np.append(x1_clean, x2_clean)

    mean = np.mean(x_clean)
    #std = np.std(x_clean, ddof=1)

    x1 -= mean
    x2 -= mean

    x1[index_x1] = 0
    x2[index_x2] = 0

    std = np.std(np.append(x1,x2), ddof=1)

    x1 = x1 / std
    x2 = x2 / std


    return x1, x2


def standardize_train(x_train):
    for i in range(x_train.shape[1]):
        x_train[:, i] = standardize_col_train(x_train[:, i])
    return x_train


def standardize_col_train(x1):
    index_x1 = np.where(x1 == -999)
    x1[index_x1] = 0
    mean = np.mean(x1)
    x1 -= mean
    x1[index_x1] = 0
    std = np.std(x1, ddof=1)
    # std = np.std(x1)
    x1 = x1 / std

    return x1


def remove_bad_data(x, y):
    # tmp = x[np.where(x != -999)]
    index = np.where(x == -999)
    index = np.unique(index[0])
    x = np.array(np.delete(x, index, 0))
    y = np.array(np.delete(y, index))
    return x, y


def remove_bad_data_test(x, y):
    # tmp = x[np.where(x != -999)]
    index = np.where(x == -999)
    index = np.unique(index[0])
    x = np.delete(x, index, 0)
    y = np.delete(y, index)
    return x, y


def remove_good_data(x, y):
    index = np.where(x != -999)
    x = np.delete(x, index[0], 0)
    y = np.delete(y, index[0], 0)
    return x, y


def discard_outliers(x_train, ys_train, threshold):
    index = []
    for i in range(x_train.shape[0]):
        if np.amax(np.abs(x_train[i, :])) > threshold:
            index.append(i)
    x_train = np.delete(x_train, index, 0)
    ys_train = np.delete(ys_train, index, 0)
    #print(len(index))
    return x_train, ys_train


def change_y_to_0(y):
    index = np.where(y == -1)
    y[index] = 0
    return y


def change_y_to_1(y):
    index = np.where(y == 0)
    y[index] = -1
    return y
