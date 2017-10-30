import matplotlib.pyplot as plt


def cross_validation_visualization_degree(degree, error_tr, error_te, plot_name):
    """
    visualization the curves of mse_tr and mse_te.

    :param degree: List of degrees to be tested
    :param error_tr: List of train error values for the different tested degrees
    :param error_te: List of test error values for the different tested degrees
    :param plot_name: Name of the generated plot
    :return: Generates a plot with the given values which is stored in the out folder.

    """

    plt.plot(degree, error_tr, marker=".", label='train error ')
    plt.plot(degree, error_te, marker=".", label='test error ')
    plt.xlabel("DEGREE")
    plt.ylabel("LOSS")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("../out/{}".format(plot_name), dpi=512)
    plt.close()


def cross_validation_visualization_lambdas(lambdas, error_tr, error_te, plot_name):
    """
    visualization the curves of mse_tr and mse_te.

    :param lambdas: List of lambdas to be tested
    :param error_tr: List of train error values for the different tested degrees
    :param error_te: List of test error values for the different tested degrees
    :param plot_name: Name of the generated plot
    :return: Generates a plot with the given values which is stored in the out folder.

    """
    plt.semilogx(lambdas, error_tr, marker=".", label='train error ')
    plt.semilogx(lambdas, error_te, marker=".", label='test error ')
    plt.xlabel("LAMBDAS")
    plt.ylabel("LOSS")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("../out/{}".format(plot_name), dpi=512)
    plt.close()

