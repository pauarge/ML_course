import numpy as np


def change_y_to_0(y):
    index = np.where(y == -1)
    y[index] = 0
    return y


def change_y_to_1(y):
    index = np.where(y == 0)
    y[index] = -1
    return y


def discard_outliers(x_train, ys_train, threshold):
    index = []
    for i in range(x_train.shape[0]):
        if np.amax(np.abs(x_train[i, :])) > threshold:
            index.append(i)
    x_train = np.delete(x_train, index, 0)
    ys_train = np.delete(ys_train, index, 0)
    return x_train, ys_train


def look_for_999(x):
    b_v = []
    for i in range(x.shape[1]):
        if np.min(x[:, i]) == -999:
            b_v.append(i)
    return b_v


def remove_bad_data(x, y):
    index = np.where(x == -999)
    index = np.unique(index[0])
    x = np.array(np.delete(x, index, 0))
    y = np.array(np.delete(y, index))
    return x, y


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
    x1 -= mean
    x2 -= mean
    x1[index_x1] = 0
    x2[index_x2] = 0

    std = np.std(np.append(x1, x2), ddof=1)

    x1 /= std
    x2 /= std
    return x1, x2
