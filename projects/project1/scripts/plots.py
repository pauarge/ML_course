import matplotlib.pyplot as plt


def cross_validation_visualization_degree(degree, mse_tr, mse_te, lambda_, plot_name):
    """visualization the curves of mse_tr and mse_te."""

    plt.plot(degree, mse_tr, marker=".", label='train error ')
    plt.plot(degree, mse_te, marker=".", label='test error ')
    plt.xlabel("DEGREE")
    plt.ylabel("MSE")
    # plt.title(plot_name)
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("../out/{}".format(plot_name), dpi=512)
    plt.close()


def cross_validation_visualization(lambdas, mse_tr, mse_te, degree, plot_name):
    """visualization the curves of mse_tr and mse_te."""

    plt.semilogx(lambdas, mse_tr, marker=".", label='train error ')
    plt.semilogx(lambdas, mse_te, marker=".", label='test error ')
    plt.xlabel("LAMBDAS")
    plt.ylabel("LOSS")
    # plt.title(plot_name)
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("../out/{}".format(plot_name), dpi=512)
    plt.close()


def plot_what_I_want():
    plt.ylabel("MSE")
    plt.xlabel("DEGREE")
    x = range(9, 14)

    plt.plot(x, [0.27099200159318815, 0.26851231691377875, 0.26792405184469548, 0.26797596503920451,
                 0.26685355535401173], marker="*", label="train error")
    plt.plot(x, [0.27193628031037409, 0.2695694948065081, 0.26905294988772011, 0.3565175890853966,
                 0.2685674638807643], marker="*", label="test error")

    # 3.732124727798658
    # 0.3879958310286602
    # plt.xticks(x, ["raw data", "substitution of\n-999 by the mean(*)", "remove\noutliers (*)",
    #               "remove\n'-999' columns (*)",
    #               "remove\n'-999' rows (*)"], fontsize=8)

    # plt.title("LOG. REGRESSION DATA PRE-PROCESS TEST LOSS COMPARISON")
    plt.grid(True)
    plt.legend(loc=2)
    plt.savefig("../out/LEAST_SQUARES_DEGREES_SUPERNICE", dpi=512)

    plt.close()


def main():
    plot_what_I_want()


if __name__ == "__main__":
    main()
