import numpy as np

from helpers import compute_gradient, compute_mse, batch_iter, sigmoid, calculate_loss, calculate_hessian, \
    calculate_gradient


def least_squares_gd(y, tx, w, max_iters, gamma):
    """Gradient descent algorithm."""
    for i in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w -= gamma * gradient
    return w, compute_mse(y, tx, w)


def least_squares_sgd(y, tx, w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    gradient = None
    for i in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
        w -= gamma * gradient
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """calculate the least squares solution.
        returns mse, and optimal weights"""
    txt = np.transpose(tx)
    w = np.linalg.solve(txt.dot(tx), txt.dot(y))
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    txt = np.transpose(tx)
    w = np.linalg.solve((txt.dot(tx) + lambda_ * 2 * y.shape[0] * np.identity(tx.shape[1])), txt.dot(y))
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    print("CALCULATING GRADIENT")
    s = sigmoid(tx.dot(w))
    # S_matrix = np.diag(sn)
    gradient = np.transpose(tx).dot(s - y)
    loss, hessian = calculate_loss(y, tx, w), calculate_hessian(tx, s)

    gamma = 0.0001
    a = np.linalg.solve(hessian, gradient)
    w = w - gamma * a
    return w, loss


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    gradient = calculate_gradient(y, tx, w)
    loss = calculate_loss(y, tx, w)
    w = w - gamma * gradient
    return w, loss


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
    return loss, w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-12
    losses = []

    w = initial_w

    # start the logistic regression
    for i in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if i % 10 == 0:
            print("Current iteration={}, loss={}".format(i, loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]
