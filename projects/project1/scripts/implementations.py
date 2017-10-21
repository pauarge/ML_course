import numpy as np

"""
    Helpers
"""


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    k = -1 / y.shape[0]
    e = y - tx.dot(w)
    return k * np.transpose(tx).dot(e)


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    k = 1 / (2 * y.shape[0])
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
    etox = np.exp(t)
    return etox / (etox + 1)


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    n = len(y)
    s = sigmoid(tx.dot(w))
    a = y * np.log(s)
    b = (np.ones(n) - y) * np.log(np.ones(n) - s)
    return -np.sum(a + b)


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.transpose(tx).dot(sigmoid(tx.dot(w)) - y)


"""
    Functions
"""


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}, gradient={g}".format(
        #    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1], g=compute_gradient(y, tx, w)))
    return ws, losses


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        # print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}, gradient={g}".format(
        #    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1], g=compute_gradient(y, tx, w)))
    return ws, losses


def least_squares(y, tx):
    """calculate the least squares solution.
        returns mse, and optimal weights"""
    w = np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))
    # w = np.linalg.inv(np.transpose(tx).dot(tx)).dot(np.transpose(tx)).dot(y)
    mse = compute_loss(y, tx, w)
    return w, mse


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    w = np.linalg.solve((np.transpose(tx).dot(tx) + lambda_ * 2 * y.shape[0] * np.identity(tx.shape[1])),
                        np.transpose(tx).dot(y))
    mse = compute_loss(y, tx, w)
    return w, mse


def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    return calculate_loss(y, tx, w), calculate_gradient(y, tx, w), calculate_hessian(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    raise NotImplemented
