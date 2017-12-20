import matplotlib.pyplot as plt
import numpy as np
import argparse

from matplotlib.ticker import MaxNLocator


from parsers import OUT_DIR

def parse_args():
    """
    Sets up a parser for CLI options.

    :return: arguments list
    """
    parser = argparse.ArgumentParser(description='Plot generation.')
    parser.add_argument('plot', type=str, default='FEA', help="Plot to generate. Options: N_EPOCHS, FEA, REG")
    parser.add_argument('title', type=bool, default=False, help="Set plot title")
    return parser.parse_args(), parser


def cross_validation_epochs(n_epochs, error_tr, error_te, plot_name, title = False):
    """
    visualization of the evolution of rmse_tr and rmse_te along the number of iterations.

    :param n_epochs: List of full passes through the full data
    :param error_tr: List of train error values for the different number of epochs
    :param error_te: List of test error values for the different number of epochs
    :param plot_name: Name of the generated plot
    :return: Generates a plot with the given values which is stored in the out folder.

    """

    plt.plot(n_epochs, error_tr, marker=".", label='train error ')
    plt.plot(n_epochs, error_te, marker=".", label='test error ')
    if title:
        plt.title("RMSE vs. Number of Epochs")
    plt.xlabel("N_EPOCHS")
    plt.ylabel("RMSE")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("../out/{}".format(plot_name), dpi=512)
    plt.close()

def cross_validation_features(k, error_tr, error_te, plot_name, title = False):
    """
    visualization of the evolution of rmse_tr and rmse_te for different number of features.

    :param k: List of features considered
    :param error_tr: List of train error values for the different number of epochs
    :param error_te: List of test error values for the different number of epochs
    :param plot_name: Name of the generated plot
    :return: Generates a plot with the given values which is stored in the out folder.

    """
    fig, (ax1, ax2) = plt.subplots(nrows = 2)
    ax1.plot(k, error_tr, marker=".", label='train error ')
    ax2.plot(k, error_te, marker=".", label='test error ')
    if title:
        plt.title("RMSE vs. Number of Features")
    plt.xlabel("Number of Features")
    ax1.set_ylabel('RMSE')
    ax2.set_ylabel('RMSE')
    ax1.legend(loc=2)
    ax2.legend(loc=2)
    ax1.grid(True)
    ax2.grid(True)
    plt.savefig("../out/{}".format(plot_name), dpi=512)
    plt.close()


def matrix_color(lambda_user, lambda_item, error_te, title = False):
    """
    visualization of the evolution of the test error (rmse) for different combinations of regularization terms.

    :param lambda_user: Numpy array of regularization terms used for user features
    :param lambda_item: Numpy array list of regularization terms used for item features
    :param error_te: Numpy array of test error values for different lambdas configurations
    :return: Generates a contour plot with the given values which is stored in the out folder

    """

    x = lambda_user
    y = lambda_item
    z = error_te

    levels = MaxNLocator(nbins=20).tick_values(z.min(), z.max())

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('YlOrRd')

    fig, (ax0) = plt.subplots(nrows=1)

    # contours are *point* based plots, so convert our bound into point
    # centers
    cf = ax0.contourf(x, y, z, levels=levels, cmap=cmap)
    fig.colorbar(cf, ax=ax0)
    if title:
        ax0.set_title('Test error vs. Regularization terms \n Bias = False')
    ax0.loglog(x, y)
    ax0.set_xlabel('Lambda User')
    ax0.set_ylabel('Lambda Item')

    # adjust spacing between subplots so `ax1` title and `ax0` tick labels
    # don't overlap
    fig.tight_layout()

    # plt.show()
    plt.savefig("{}/{}".format(OUT_DIR, "lambdas_bias"), dpi=512)
    plt.close()



def main():
    args, parser = parse_args()
    title = args.title

    if args.plot == 'N_EPOCHS':
        n_epochs = [1, 2, 5, 10, 15, 20, 25, 50, 75, 125, 150, 250, 500]
        error_te = [1.919213, 1.549221, 1.749835, 1.440623, 1.444905, 1.278618, 1.262399, 1.036650,
                    1.02321, 1.007170, 1.008266, 1.004835, 1.003473, 1.002028]
        error_tr = [1.913701, 1.547222, 1.739495, 1.429137, 1.421281, 1.248655, 1.227077, 0.968838,
                    0.9472955, 0.9156492, 0.906408, 0.893195, 0.868574, 0.848911]
        plot_name = "n_epochs"
        cross_validation_epochs(n_epochs, error_tr, error_te, plot_name, title)

    elif args.plot == 'FEA':
        k = [10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100, 105, 110, 120]
        error_tr = [0.968368, 0.954892, 0.945716, 0.941496, 0.939122, 0.936870, 0.935327, 0.934809, 0.934411, 0.934378,
                    0.934024,
                    0.934253, 0.933864, 0.934197, 0.933947]
        error_te = [1.001359, 1.000508, 0.998801, 0.998669, 0.998390, 0.997850, 0.997356, 0.997542, 0.997029, 0.996990,
                    0.996763,
                    0.996995, 0.996853, 0.996906, 0.996604]
        plot_name = "RMSE_features"
        cross_validation_features(k, error_tr, error_te, plot_name, title)

    elif args.plot == 'REG':
        lambda_user = np.array([[0.001, 0.01, 0.1], [0.001, 0.01, 0.1], [0.001, 0.01, 0.1]])
        lambda_item = np.array([[0.001, 0.001, 0.001], [0.01, 0.01, 0.01], [0.1, 0.1, 0.1]])
        error_te = np.array([[1.0638, 1.0551, 1.0371], [1.0571, 1.0482, 1.0203], [1.0476, 1.0257, 1.0012]])
        matrix_color(lambda_user, lambda_item, error_te, title)


if __name__ == '__main__':
    main()
