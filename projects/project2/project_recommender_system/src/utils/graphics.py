import matplotlib.pyplot as plt

def cross_validation_epochs(n_epochs, error_tr, error_te, plot_name):
    """
    visualization the curves of mse_tr and mse_te.

    :param degree: List of degrees to be tested
    :param error_tr: List of train error values for the different tested degrees
    :param error_te: List of test error values for the different tested degrees
    :param plot_name: Name of the generated plot
    :return: Generates a plot with the given values which is stored in the out folder.

    """

    plt.plot(n_epochs, error_tr, marker=".", label='train error ')
    plt.plot(n_epochs, error_te, marker=".", label='test error ')
    plt.xlabel("N_EPOCHS")
    plt.ylabel("RMSE")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("../out/{}".format(plot_name), dpi=512)
    plt.close()


