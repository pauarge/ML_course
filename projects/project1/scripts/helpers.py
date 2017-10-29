import numpy as np


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches

    :param y: Vector of labels of size 1xN
    :param tx: Matrix of input data of size Nx(1+(DG*D)) after adding a column of ones
    :param batch_size: # data points used included in the batch
    :param num_batches: Number of batches to be generated
    :param shuffle: Boolean whether to randomly order the initial data
    :return: Iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`

    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param x: Matrix of input variables of size (NxD)
    :param degree: Degree of the resulting polynomial
    :return: Matrix of the polynomial basis functions
    """
    tx = np.ones(x.shape[0])
    for i in range(1, degree + 1):
        tx = np.column_stack((tx, np.power(x, i)))
    return tx


def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold
    N = #data points
    D = #number of variables in input data

    :param y: Vector of labels of size 1xN
    :param k_fold: Number of folds in which data is split
    :param seed: Number used to initialize a pseudorandom number generator.
    :return: k_fold vectors of size (1xN/k_fold) in which indices of input data are saved
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def compute_gradient(y, tx, w):
    """
    Compute the gradient
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param y: Vector of labels of size 1xN
    :param tx: Matrix of input data of size Nx(1+(DG*D)) after adding a column of ones
    :param w: Vector of weights of size 1x(1+(DG*D))
    :return: Gradient for loss function of Mean Squared Error evaluated in w
    """
    k = -1.0 / y.shape[0]
    y_pred = predict_labels(w, tx)
    e = y - y_pred
    return k * np.transpose(tx).dot(e)


def compute_mse(y, tx, w):
    """
    Calculate the loss using MSE
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param y: Vector of labels of size 1xN
    :param tx: Matrix of input data of size Nx(1+(DG*D)) after adding a column of ones
    :param w: Vector of weights of size 1x(1+(DG*D))
    :return: Loss value for Mean Squared Error evaluated in w
    """
    k = 1.0 / (2 * y.shape[0])
    y_pred = tx.dot(w)
    e = y - y_pred
    return k * np.transpose(e).dot(e)


def calculate_loss(y, tx, w):
    """
    Compute the cost by negative log likelihood
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param y: Vector of labels of size 1xN
    :param tx: Matrix of input data of size Nx(1+(DG*D)) after adding a column of ones
    :param w: Vector of weights of size 1x(1+(DG*D))
    :return: Loss value by negative log likelihood evaluated in w
    """
    n = len(y)
    y_pred = new_labels(w, tx)
    s = sigmoid(y_pred)
    a = np.log(s) * y
    o = np.ones(n)
    b = (o - np.transpose(y)) * np.log(o - np.transpose(s))
    return (-np.sum(a + np.transpose(b))) / tx.shape[0]


def calculate_loss_reg(y, tx, w, lambda_):
    """
    Compute the cost by negative log likelihood for Regularized Logistic Regression
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param y: Vector of labels of size 1xN
    :param tx: Matrix of input data of size Nx(1+(DG*D)) after adding a column of ones
    :param w: Vector of weights of size 1x(1+(DG*D))
    :param lambda_: Regularization parameter
    :return: Loss value by negative log likelihood for Regularized Logistic Regression evaluated in w
    """
    n = tx.shape[0]
    return calculate_loss(y, tx, w) + (lambda_ / (2 * n)) * np.power(np.linalg.norm(w), 2)


def calculate_gradient(y, tx, w):
    """
    Compute the gradient of the negative log likelihood loss function
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param y: Vector of labels of size 1xN
    :param tx: Matrix of input data of size Nx(1+(DG*D)) after adding a column of ones
    :param w: Vector of weights of size 1x(1+(DG*D))
    :return: Gradient of the negative log likelihood loss function in w
    """
    y_pred = new_labels(w, tx)
    s = sigmoid(y_pred)
    k = 1.0 / y.shape[0]
    return k * np.transpose(tx).dot(s - y)


def sigmoid(t):
    """
    Apply sigmoid function on t

    :param t: Vector in which sigmoid is evaluated
    :return: Sigmoid function evaluated in t
    """
    return 1 / (1 + np.exp(-t))


def predict_labels(weights, data):
    """
    Generates class predictions given weights, and a test data matrix for Least Squares
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param weights: Vector of weights of size 1x(1+(DG*D))
    :param data: Matrix containing the test data
    :return: Class predictions for given weights and a test data matrix for Least Squares
    """
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred


def predict_labels_log(weights, data):
    """
    Generates class predictions given weights, and a test data matrix for Logistic Regression
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param weights: Vector of weights of size 1x(1+(DG*D))
    :param data: Matrix containing the test data
    :return: Class predictions for given weights and test data matrix for Logistic Regression
    """
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred


def new_labels(w, tx):
    """
    Generates class predictions given weights, and a test data matrix
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param w: Vector of weights of size 1x(1+(DG*D))
    :param tx: Matrix of input data of size Nx(1+(DG*D)) after adding a column of ones
    :return: Class predictions given weights, and a test data matrix
    """
    y_pred = tx.dot(w)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param y: Vector of labels of size 1xN
    :param tx: Matrix of input data of size Nx(1+(DG*D)) after adding a column of ones
    :param w: Vector of initial weights of size 1x(1+(DG*D))
    :param gamma: Step size of the iterative method
    :param lambda_: Regularization parameter
    :return: Loss and updated w after doing one step of gradient descent for penalized logistic regression

    """
    loss = calculate_loss_reg(y, tx, w, lambda_)
    gradient = calculate_gradient(y, tx, w) + (1 / tx.shape[0]) * lambda_ * w
    w = w - gamma * gradient
    return w, loss, np.linalg.norm(gradient)
