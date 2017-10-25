import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    tx = np.ones(x.shape[0])
    for i in range(1, degree + 1):
        tx = np.column_stack((tx, np.power(x, i)))
    return tx


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    k = -1.0 / y.shape[0]
    e = y - tx.dot(w)
    return k * np.transpose(tx).dot(e)


def compute_mse(y, tx, w):
    """Calculate the loss.
        Using MSE
    """
    k = 1.0 / (2 * y.shape[0])
    e = y - tx.dot(w)
    return k * np.transpose(e).dot(e)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
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


def sigmoid(t):
    """apply sigmoid function on t."""
    # print("CALCULATING SIGMOID")
    return 1 / (1 + np.exp(-t))


# def calculate_loss(y, tx, w):
#     """compute the cost by negative log likelihood."""
#     print("CALCULATING LOSS")
#     y_pred = tx.dot(w)
#     a = np.exp(y_pred)
#     b = np.ones(y.shape[0])
#     c = np.log(a + b)
#     d = y * y_pred
#     e = c - d
#     return np.sum(e)

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""

    n = len(y)
    y_pred = new_labels(w, tx)
    s = sigmoid(y_pred)
    aa = np.log(s)
    a = aa * y
    os = np.ones(n)
    r = os - np.transpose(s)
    rlog = np.log(r)
    j = np.ones(n) - np.transpose(y)
    b = j * rlog
    return (-np.sum(a + np.transpose(b))) / y.shape[0]


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    y_pred = new_labels(w, tx)
    s = sigmoid(y_pred)
    k = 1.0 / y.shape[0]
    return k * np.transpose(tx).dot(s - y)


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred


def new_labels(w, tx):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = tx.dot(w)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    n = tx.shape[0]
    loss = calculate_loss(y, tx, w) + (lambda_ / (2 * n)) * np.power(np.linalg.norm(w), 2)
    gradient = calculate_gradient(y, tx, w) + (1 / n) * lambda_ * w
    # print("GRADIENT {}: ".format(gradient))

    # Quan volguem afegir hessian tb caldra dividir per N
    # a = np.eye(w.shape[0], dtype=float) * lambda_
    # hessian = calculate_hessian(tx, sigmoid(tx)) + a
    return loss, gradient


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)

    # a = np.linalg.solve(hessian, gradient)
    # w = w - gamma * a
    w = w - gamma * gradient
    return w, loss
