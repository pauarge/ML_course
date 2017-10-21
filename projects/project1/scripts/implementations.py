from helpers import *


def least_squares_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w -= gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}, gradient={g}".format(
        #    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1], g=compute_gradient(y, tx, w)))
    return ws[-1], losses[-1]


def least_squares_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    gradient = None
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        # print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}, gradient={g}".format(
        #    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1], g=compute_gradient(y, tx, w)))
    return ws[-1], losses[-1]


def least_squares(y, tx):
    """calculate the least squares solution.
        returns mse, and optimal weights"""
    w = np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))
    mse = compute_mse(y, tx, w)
    return w, mse


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    w = np.linalg.solve((np.transpose(tx).dot(tx) + lambda_ * 2 * y.shape[0] * np.identity(tx.shape[1])),
                        np.transpose(tx).dot(y))
    mse = compute_mse(y, tx, w)
    return w, mse


def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    return calculate_loss(y, tx, w), calculate_gradient(y, tx, w), calculate_hessian(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    raise NotImplemented
