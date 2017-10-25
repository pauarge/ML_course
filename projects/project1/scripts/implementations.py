from helpers import *


def least_squares_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    loss = None
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w -= gamma * gradient
    return w, loss


def least_squares_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    loss = None
    gradient = None
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
        loss = compute_mse(y, tx, w)
        w -= gamma * gradient
    return w, loss


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
    s = sigmoid(tx.dot(w))
    # S_matrix = np.diag(sn)
    print("CALCULATING GRADIENT")
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


def logistic_regression_newton(y, tx, w):
    # init parameters
    max_iter = 10
    threshold = 1e-8
    losses = []

    # w = np.zeros(tx.shape[1])

    # start the logistic regression
    print("LOGISTIC REGRESSION NEWTON")
    for i in range(max_iter):
        print("iteracio{}.format(i)")
        # get loss and update w.
        w, loss = logistic_regression(y, tx, w)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, calculate_loss(y, sigmoid(tx.dot(w)))


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y, tx, w) + (lambda_ / 2) * np.power(np.linalg.norm(w), 2)
    gradient = calculate_gradient(y, tx, w) + lambda_ * w
    print("GRADIENT {}: ".format(gradient))

    a = np.eye(w.shape[0], dtype=float) * lambda_
    hessian = calculate_hessian(tx, sigmoid(tx)) + a
    return loss, gradient, hessian


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)

    a = np.linalg.solve(hessian, gradient)
    w = w - gamma * a
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
        if i % 1 == 0:
            print("Current iteration={}, loss={}".format(i, loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    print(losses)

    return w, losses[-1]
