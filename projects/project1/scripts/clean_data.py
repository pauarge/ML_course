import numpy as np


def change_y_to_0(y):
    """
    Modify label values from -1 to 0
    N = #data points

    :param y: Array of labels of size 1xN
    :return: Array of labels with values 0 or 1
    """
    index = np.where(y == -1)
    y[index] = 0
    return y


def change_y_to_1(y):
    """
    Modify label values from 0 to -1
    N = #data points

    :param y: Array of labels of size 1xN
    :return: Array of labels with values -1 or 1
    """
    index = np.where(y == 0)
    y[index] = -1
    return y


def discard_outliers(x_train, ys_train, threshold):
    """
    Discards data points containing outliers as a coordinate (coordinates which have values far from the mean of that column)
    N = #data points
    D = #number of variables in input data

    :param x_train: Matrix of input variables of size NxD
    :param ys_train: Vector of labels of size 1xN
    :param threshold: Value indicating the presence of outliers
    :return: Train data without outliers
    """
    index = []
    for i in range(x_train.shape[0]):
        if np.amax(np.abs(x_train[i, :])) > threshold:
            index.append(i)
    x_train = np.delete(x_train, index, 0)
    ys_train = np.delete(ys_train, index, 0)
    return x_train, ys_train


def look_for_999(x):
    """
    Return an array with the columns that have -999 values
    N = #data points
    D = #number of variables in input data

    :param x: Matrix of input variables of size NxD
    :return: Array with the columns that have -999 values
    """
    b_v = []
    for i in range(x.shape[1]):
        if np.min(x[:, i]) == -999:
            b_v.append(i)
    return b_v


def remove_bad_data(x, y):
    """
    Deletes data points containing -999 values
    N = #data points
    D = #number of variables in input data

    :param x: Matrix of input variables of size NxD
    :param y: Vector of labels of size 1xN
    :return: Data points without input outliers
    """
    index = np.where(x == -999)
    index = np.unique(index[0])
    x = np.array(np.delete(x, index, 0))
    y = np.array(np.delete(y, index))
    return x, y


def standardize(x_test, x_train):
    """
    Standardizes each column of the train and test data points
    N1 = #train data points
    N2 = #test data points
    D = #number of variables in input data

    :param x_test: Matrix of test data points of size N1xD
    :param x_train: Train data points N2xD
    :return: Standardized matrices x_test and x_train
    """
    for i in range(x_test.shape[1]):
        x_test[:, i], x_train[:, i] = standardize_col(x_test[:, i], x_train[:, i])
    return x_test, x_train


def standardize_col(x1, x2):
    """
    Standardizes arrays x1 and x2 of the train and test data
    N1 = #train data points
    N2 = #test data points

    :param x1: Array of the train data of size N1x1
    :param x2: Array of the test data of size N2x1
    :return: Standardized arrays x1 and x2
    """
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
