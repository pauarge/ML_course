import numpy as np

from helpers import compute_gradient, compute_mse, batch_iter, calculate_loss, calculate_gradient, \
    learning_by_penalized_gradient


def least_squares_gd(y, tx, w, max_iters, gamma):
    """
    Linear regression using gradient descent
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param y: Vector of labels of size 1xN
    :param tx: Matrix of input data of size Nx(1+(DG*D)) after adding a column of ones
    :param w: Vector of initial weights of size 1x(1+(DG*D))
    :param max_iters: Number of maximum iterations on the loop
    :param gamma: Step size of the iterative method
    :return: Vector of weights of size 1x(1+(DG*D)) and
             Mean Squared Error of the obtained weights
    """
    error = []
    for i in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w -= gamma * gradient
        mse = compute_mse(y, tx, w)
        error.append(mse)
        if len(error) > 100 and np.abs(error[-1] - error[-2]) < THRESHOLD:
            gamma = gamma / 10
            if gamma < 1e-10:
                break

    return w, error[-1]


def least_squares_sgd(y, tx, w, batch_size, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param y: Vector of labels of size 1xN
    :param tx: Matrix of input data of size Nx(1+(DG*D)) after adding a column of ones
    :param w: Vector of initial weights of size 1x(1+(DG*D))
    :param batch_size: Amount of data points taken on each batch of the iteration
    :param max_iters: Number of maximum iterations on the loop
    :param gamma: Step size of the iterative method
    :return: Vector of weights of size 1x(1+(DG*D)) and
             Mean Squared Error of the obtained weights
    """
    gradient = None
    error = []
    for i in range(max_iters):

        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
        w -= gamma * gradient
        mse = compute_mse(y, tx, w)
        error.append(mse)
        if len(error) > 100 and np.abs(error[-1] - error[-2]) < THRESHOLD:
            gamma = gamma / 10
            if gamma < 1e-10:
                break
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """
    Least squares regression using normal equations
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param y: Vector of labels of the input data of size 1xN
    :param tx: Matrix of input variables of size Nx(1+(DG*D)) after adding a column of ones
    :return: Vector of weights of size 1x(1+(DG*D)) and
             Mean Squared Error of the obtained weights
    """
    txt = np.transpose(tx)
    w = np.linalg.solve(txt.dot(tx), txt.dot(y))
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param y: Vector of labels of size 1xN
    :param tx: Matrix of input variables of size Nx(1+(DG*D)) after adding a column of ones
    :param lambda_: Regularization parameter
    :return: Vector of weights of size 1x(1+(DG*D)) and
             Mean Squared Error of the obtained weights
    """
    txt = np.transpose(tx)
    w = np.linalg.solve((txt.dot(tx) + lambda_ * 2 * y.shape[0] * np.identity(tx.shape[1])), txt.dot(y))
    return w, compute_mse(y, tx, w)


# Threshold condition for stopping the iterations on logistic regression and regularized logistic regression
THRESHOLD = 1e-6


def logistic_regression(y, tx, w, max_iters, gamma):
    """
    Logistic regression using gradient descent
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param y: Vector of labels of size 1xN
    :param tx: Matrix of input variables of size Nx(1+(DG*D)) after adding a column of ones
    :param w: Vector of initial weights of size 1x(1+(DG*D))
    :param max_iters: Number of maximum iterations on the loop
    :param gamma: Step size of the iterative method
    :return: Vector of weights of size 1x(1+(DG*D)) and
             loss of the weights computed by negative log likelihood
    """
    losses = []

    for i in range(max_iters):
        gradient = calculate_gradient(y, tx, w)
        loss = calculate_loss(y, tx, w)
        w = w - gamma * gradient
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < THRESHOLD:
            gamma = gamma / 10
            if gamma < 1e-10:
                break
        if len(losses) > 100 and losses[-1] > losses[-100]:
            gamma = gamma / 10

    return w, losses[-1]


def reg_logistic_regression(y, tx, lambda_, w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent
    N = #data points
    D = #number of variables in input data
    DG = Degree of the polynomial

    :param y: Vector of labels of size 1xN
    :param tx: Matrix of input variables of size Nx(1+(DG*D)) after adding a column of ones
    :param lambda_: Regularization parameter
    :param w: Vector of initial weights of size 1x(1+(DG*D))
    :param max_iters: Number of maximum iterations on the loop
    :param gamma: Step size of the iterative method
    :return: Vector of weights of size 1x(1+(DG*D)) and
             loss of the weights computed by negative log likelihood
    """
    losses = []
    for i in range(max_iters):
        w, loss, grad_norm = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        losses.append(loss)
        if len(losses) > 100 and np.abs(losses[-1] - losses[-100]) < THRESHOLD:
            gamma = gamma / 10
            if gamma < 1e-10:
                break
        if len(losses) > 100 and losses[-1] > losses[-100]:
            gamma = gamma / 10
    return w, losses[-1]
