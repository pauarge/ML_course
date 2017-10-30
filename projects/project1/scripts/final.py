from clean_data import standardize, discard_outliers
from helpers import build_poly
from implementations import least_squares_gd, least_squares_sgd, least_squares
from parsers import load_data

OUT_DIR = "../out"
OUTLIERS_THRESHOLD = 9.2
DEGREE = 2


def main():
    ys_train, x_train, ids_train, x_test, ids_test = load_data()

    print("FILTERING DATA")
    x_test, x_train = standardize(x_test, x_train)
    x_train, ys_train = discard_outliers(x_train, ys_train, OUTLIERS_THRESHOLD)

    print("BUILDING POLYNOMIALS")
    tx_train = build_poly(x_train, DEGREE)

    print("LEARNING MODEL BY LEAST SQUARES")
    w_ls, mse_ls = least_squares(ys_train, tx_train)
    print(mse_ls)

    w_ini = w_ls
    print("LEARNING MODEL BY GRADIENT DESCENT")
    max_iters = 5000
    gamma = 0.01
    w_gd, mse_gd = least_squares_gd(ys_train, tx_train, w_ini, max_iters, gamma)
    print(mse_gd)

    print("LEARNING MODEL BY STOCHASTIC GRADIENT DESCENT")
    max_iters = 5000
    gamma = 0.01
    batch_size = 10000
    w_sgd, mse_sgd = least_squares_sgd(ys_train, tx_train, w_ini, batch_size, max_iters, gamma)
    print(mse_sgd)

    print("LEAST SQUARES\n W: {} MSE:{}".format(w_ls, mse_ls))
    print("GRAD DESCENT\n W: {} MSE:{}".format(w_gd, mse_gd))
    print("STOC GRAD DESCENT\n W: {} MSE:{}".format(w_sgd, mse_sgd))


if __name__ == '__main__':
    main()
