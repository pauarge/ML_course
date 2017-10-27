import numpy as np

from helpers import compute_gradient, compute_mse, batch_iter, calculate_loss, calculate_gradient, \
    learning_by_penalized_gradient

THRESHOLD = 1e-12


def least_squares_gd(y, tx, w, max_iters, gamma):
    """
    Linear regression using gradient descent

    :param y:
    :param tx:
    :param w:
    :param max_iters:
    :param gamma:
    :return:
    """
    for i in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w -= gamma * gradient
    return w, compute_mse(y, tx, w)


def least_squares_sgd(y, tx, w, batch_size, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent

    :param y:
    :param tx:
    :param w:
    :param batch_size:
    :param max_iters:
    :param gamma:
    :return:
    """
    """Stochastic gradient descent algorithm."""
    gradient = None
    for i in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
        w -= gamma * gradient
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """
    Least squares regression using normal equations

    :param y:
    :param tx:
    :return:
    """
    txt = np.transpose(tx)
    w = np.linalg.solve(txt.dot(tx), txt.dot(y))
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations

    :param y:
    :param tx:
    :param lambda_:
    :return:
    """
    txt = np.transpose(tx)
    w = np.linalg.solve((txt.dot(tx) + lambda_ * 2 * y.shape[0] * np.identity(tx.shape[1])), txt.dot(y))
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, w, max_iters, gamma):
    """
    Logistic regression using gradient descent

    :param y:
    :param tx:
    :param w:
    :param max_iters:
    :param gamma:
    :return:
    """
    losses = []

    for i in range(max_iters):
        gradient = calculate_gradient(y, tx, w)
        loss = calculate_loss(y, tx, w)
        w = w - gamma * gradient
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < THRESHOLD:
            break

    return w, losses[-1]


def reg_logistic_regression(y, tx, lambda_,  max_iters, gamma):
    """
    Regularized logistic regression using gradient descent

    :param y:
    :param tx:
    :param lambda_:
    :param w:
    :param max_iters:
    :param gamma:
    :return:
    """
    losses = []
    w, _= least_squares(y,tx)
    loss_ls = calculate_loss(y,tx,w)
    #print("LEAST SQ LOSS{}".format(loss_ls))
    for i in range(max_iters):
        w, loss, grad_norm = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        if i % 50 == 0:
            print("Current iteration={}, norm_grad={}, gamma={}".format(i, grad_norm, gamma))

        losses.append(loss)
        #if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < THRESHOLD:
        #    print("out for threshold")
        #    break

        if len(losses)> 100 and losses[-1] > losses[-100]:
            gamma = gamma/10

    return w, losses[-1]
