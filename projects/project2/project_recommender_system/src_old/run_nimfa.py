import numpy as np
import nimfa

from parse_nimfa import create_submission


def run():
    """
    Run SNMF/R on the MovieLens data set.
    
    Factorization is run on `ua.base`, `ua.test` and `ub.base`, `ub.test` data set. This is MovieLens's data set split 
    of the data into training and test set. Both test data sets are disjoint and with exactly 10 ratings per user
    in the test set. 
    """
    V = read("data/data_train.csv.train.txt")
    W, H = factorize(V)
    rmse(W, H, "data/data_train.csv.test.txt")


def factorize(V):
    """
    Perform SNMF/R factorization on the sparse MovieLens data matrix. 
    
    Return basis and mixture matrices of the fitted factorization model. 
    
    :param V: The MovieLens data matrix. 
    :type V: `numpy.matrix`
    """
    snmf = nimfa.Snmf(V, seed="random_vcol", rank=20, max_iter=100, version='r', eta=1.,
                      beta=1e-4, i_conv=10, w_min_change=0)
    print("Algorithm: %s\nInitialization: %s\nRank: %d" % (snmf, snmf.seed, snmf.rank))
    fit = snmf()
    sparse_w, sparse_h = fit.fit.sparseness()
    print("""Stats:
            - iterations: %d
            - Euclidean distance: %5.3f
            - Sparseness basis: %5.3f, mixture: %5.3f""" % (fit.fit.n_iter,
                                                            fit.distance(metric='euclidean'),
                                                            sparse_w, sparse_h))
    return fit.basis(), fit.coef()


def read(data_set):
    """
    Read movies' ratings data from MovieLens data set. 
    
    :param data_set: Name of the split data set to be read.
    :type data_set: `str`
    """
    print("Read MovieLens data set")
    V = np.ones((10000, 1000)) * 3
    for line in open(data_set):
        u, i, r = list(map(int, line.split()))
        V[u - 1, i - 1] = r
    return V


def rmse(W, H, data_set):
    """
    Compute the RMSE error rate on MovieLens data set.
    
    :param W: Basis matrix of the fitted factorization model.
    :type W: `numpy.matrix`
    :param H: Mixture matrix of the fitted factorization model.
    :type H: `numpy.matrix`
    :param data_set: Name of the split data set to be read. 
    :type data_set: `str`
    """
    rmse = []
    for line in open(data_set):
        u, i, r = list(map(int, line.split()))
        sc = max(min((W[u - 1, :] * H[:, i - 1])[0, 0], 5), 1)
        rmse.append((sc - r) ** 2)
    print("RMSE: %5.3f" % np.sqrt(np.mean(rmse)))
    create_submission(W, H)


if __name__ == "__main__":
    """Run the Recommendations example."""
    run()
