import numpy as np
import sys

from parsers import load_pickle_data, dump_pickle_data


def main(argv):
    print("LOADING SOLUTIONS")
    sol = load_pickle_data("solutions")
    if sol is None:
        sol = np.genfromtxt("../data/solutions.csv", delimiter=",", skip_header=1, dtype=str, usecols=[0, 32])
        sol = sol[250000:]
        sol[np.where(sol == 'b')] = -1
        sol[np.where(sol == 's')] = 1
        sol = sol.astype(int)
        dump_pickle_data(sol, "solutions")

    print("LOADING INPUT")
    inp = np.genfromtxt("../out/{}.csv".format(argv[0]), delimiter=",", skip_header=1, dtype=int)

    print("CALCULATING SCORE")
    res = sol[:, 1] * inp[:, 1]
    score = round((len(np.where(res > 0)[0])) / float(inp.shape[0]), 10)

    print("\nScore: {}".format(score))


if __name__ == '__main__':
    main(sys.argv[1:])
