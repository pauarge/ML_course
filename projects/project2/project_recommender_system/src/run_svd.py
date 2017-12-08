from helpers import calculate_mse
from methods import matrix_factorization_SGD, ALS
from parsers import load_data, create_submission
from SVD import computeSVD
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg


def run(lambda_user=0.1, lambda_item=0.01, num_features=30, min_num_data=10, p_test=0.2):
    print("LOADING DATA...")
    train, test, transformation_user, transformation_item = load_data(min_num_data)

    K = 30
    U, S, Vt = sp.sparse.linalg.svd(train)

    print(U, S, Vt)


if __name__ == '__main__':
    run()
