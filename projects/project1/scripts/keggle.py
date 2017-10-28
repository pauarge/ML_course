import numpy as np
import sys


def main(argv):
    print("LOADING SOLUTIONS")
    sol = np.genfromtxt("../data/solutions.csv", delimiter=",", skip_header=1, dtype=str, usecols=[0, 32])
    sol = sol[250000:]
    sol[np.where(sol == 'b')] = -1
    sol[np.where(sol == 's')] = 1
    sol = sol.astype(int)

    print("LOADING INPUT")
    inp = np.genfromtxt("../data/{}.csv".format(argv[0]), delimiter=",", skip_header=1, dtype=int)

    print("CALCULATING SCORE")
    res = sol[:, 1] * inp[:, 1]
    score = round((len(np.where(res > 0)[0]) * 100.0) / float(inp.shape[0]), 5)

    print("\nScore: {}%".format(score))


if __name__ == '__main__':
    main(sys.argv[1:])
