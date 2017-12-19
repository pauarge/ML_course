import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator


def cross_validation_epochs(n_epochs, error_tr, error_te, plot_name):
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
    plt.xlabel("N_EPOCHS")
    plt.ylabel("RMSE")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("../out/{}".format(plot_name), dpi=512)
    plt.close()


def cross_validation_lam(lambda_user, lambda_item, error_te, plot_name):
    """
    visualization the curves of mse_tr and mse_te.

    :param degree: List of degrees to be tested
    :param error_tr: List of train error values for the different tested degrees
    :param error_te: List of test error values for the different tested degrees
    :param plot_name: Name of the generated plot
    :return: Generates a plot with the given values which is stored in the out folder.

    """

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    Y = np.array(lambda_user)
    X = np.array(lambda_item)
    Z = error_te

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0.99, 1.07)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.005, aspect=5)


    #plt.xlabel("LAMBDA_USER")
    #plt.ylabel("LAMBDA_ITEM")

    plt.legend(loc=2)
    plt.show()
    #plt.savefig("/{}".format(plot_name), dpi=512)
    #plt.close()



def matrix_color(lambda_user, lambda_item, error_te):
    # make these smaller to increase the resolution
    dx, dy = 0.05, 0.05

    # generate 2 2d grids for the x & y bounds
    x = np.array(lambda_user)
    y = np.array(lambda_item)
    z = error_te

    levels = MaxNLocator(nbins=20).tick_values(z.min(), z.max())

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('PiYG')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, (ax1) = plt.subplots(nrows=1)
    #
    # im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    # fig.colorbar(im, ax=ax0)
    # ax0.set_title('pcolormesh with levels')

    # contours are *point* based plots, so convert our bound into point
    # centers
    cf = ax1.contourf(x,y, z, levels=levels,cmap=cmap)
    fig.colorbar(cf, ax=ax1)
    ax1.set_title('contourf with levels')

    # adjust spacing between subplots so `ax1` title and `ax0` tick labels
    # don't overlap
    fig.tight_layout()

    plt.show()




def main():
    lambda_user = [[0.001, 0.01, 0.1], [0.001, 0.01, 0.1], [0.001, 0.01, 0.1]]
    lambda_item = [[0.001, 0.001, 0.001], [0.01, 0.01, 0.01], [0.1, 0.1, 0.1]]
    error_te = np.array([[1.0638, 1.0551, 1.0371], [1.0571, 1.0482, 1.0203], [1.0476, 1.0257, 1.0012]])
    #cross_validation_lam(lambda_user, lambda_item, error_te, "LAMBDA")
    matrix_color(lambda_user, lambda_item, error_te)

if __name__ == '__main__':
    main()
