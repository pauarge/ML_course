from sparsesvd import sparsesvd
import math as mt
import numpy as np

from scipy.sparse import csr_matrix


def computeSVD(train, K):
    print("compute SVD")
    U, s, Vt = sparsesvd(train, K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i, i] = mt.sqrt(s[i])

    U = csr_matrix(np.transpose(U), dtype=np.float32)
    S = csr_matrix(S, dtype=np.float32)
    Vt = csr_matrix(Vt, dtype=np.float32)

    return U, S, Vt
