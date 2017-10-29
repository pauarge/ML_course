import numpy as np
import sys

from parsers import load_pickle_data, dump_pickle_data


def main(argv, silent=False, fullpath=False):
    if not silent:
        print("LOADING SOLUTIONS")
    sol = load_pickle_data("solutions")
    if sol is None:
        sol = np.genfromtxt("../data/solutions.csv", delimiter=",", skip_header=1, dtype=str, usecols=[0, 32])
        sol = sol[250000:]
        sol[np.where(sol == 'b')] = -1
        sol[np.where(sol == 's')] = 1
        sol = sol.astype(int)
        dump_pickle_data(sol, "solutions")

    if not silent:
        print("LOADING INPUT")
    inpfile = argv[0] if fullpath else "../out/{}.csv".format(argv[0])
    inp = np.genfromtxt(inpfile, delimiter=",", skip_header=1, dtype=int)

    if not silent:
        print("CALCULATING SCORE")
    res = sol[:, 1] * inp[:, 1]
    score = round((len(np.where(res > 0)[0])) / float(inp.shape[0]), 10)

    print("Score {}: {}".format(argv[0], score))

    return score


if __name__ == '__main__':
    main(sys.argv[1:])
