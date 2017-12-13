from utils.helpers import build_k_indices
from utils.parsers import load_data
from utils.validation import cross_validation


def main():
    ys_train, x_train, ids_train, x_test, ids_test = load_data()

    seed = 3
    k_fold = 4

    # split data in k fold
    k_indices = build_k_indices(ys_train, k_fold, seed)

    tr, te = 0, 0

    for j in range(k_fold):
        tmp_tr, tmp_te = cross_validation(ys_train, x_train, k_indices, j, lambda_=0.01)
        tr += tmp_tr
        te += tmp_te

    print("TEST ERROR {} FOR {} METHOD".format(te / k_fold, method))
    print("TRAIN ERROR {} FOR {} METHOD".format(tr / k_fold, method))


if __name__ == '__main__':
    main()
