import numpy as np
import pickle
import os


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
    print("CALCULATING SIGMOID")
    return 1 / (1 + np.exp(-t))


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    print("CALCULATING LOSS")
    y_pred = tx.dot(np.transpose(w))
    y_pred[np.where(y_pred <= 0)] = 0
    y_pred[np.where(y_pred > 0)] = 1
    a = np.exp(y_pred)
    b = np.ones(len(y))
    c = np.log(a + b)
    d = y * y_pred
    e = c - d
    return -np.sum(e)


def calculate_hessian(tx, s):
    # """return the hessian of the loss function."""
    print("CALCULATING HESSIAN")
    a = s.flatten()
    txt = np.transpose(tx)
    h = np.zeros([tx.shape[1], tx.shape[1]])
    for i in range(tx.shape[1]):
        h[i, i] = txt[i, i] * a[i] * tx[i, i]
    return h


"""
def calculate_hessian(tx, S_matrix):
    return the hessian of the loss function.
    # a = s.flatten()
    txt = np.transpose(tx)
    h = txt.dot(S_matrix).dot(tx)
    print("CALCULATING HESSIAN")
    return h

"""


# Pickle

def load_data(filename):
    path = "../tmp/{}.pckl".format(filename)
    if os.path.exists(path):
        f = open(path, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj


def dump_data(obj, filename):
    path = "../tmp/{}.pckl".format(filename)
    f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()
