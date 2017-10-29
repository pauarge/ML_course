import matplotlib.pyplot as plt


def cross_validation_visualization(lambdas, mse_tr, mse_te, degree, plot_name):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambdas, mse_tr, marker=".", label='train error {}'.format(degree))
    plt.semilogx(lambdas, mse_te, marker=".", label='test error {}'.format(degree))
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title(plot_name)
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("../out/{}".format(plot_name), dpi=512)
    plt.close()


def plot_what_I_want():
    plt.ylabel("TEST MSE")
    x = [0, 1.5, 3, 4.5, 6]
    plt.plot(x, [0.33957, 0.34051, 0.26918, 0.29433, 0.42879], 'ro')
    plt.xticks(x, ["raw data", "substitution of\n-999 by the mean(*)", "remove\noutliers (*)",
                   "remove\n'-999' columns (*)", \
                   "remove\n'-999' rows (*)"], fontsize=8)

    plt.title("LEAST SQUARES DATA PRE-PROCESS TEST MSE COMPARISON")
    plt.grid(True)
    plt.savefig("../out/LEAST_SQUARES_DATA_PRE-PROCESS", dpi=512)

    plt.close()


def main():
    plot_what_I_want()


if __name__ == "__main__":
    main()
