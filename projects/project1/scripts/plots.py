import matplotlib.pyplot as plt


def cross_validation_visualization(lambds, mse_tr, mse_te, degree, plot_name):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", label='train error {}'.format(degree))
    plt.semilogx(lambds, mse_te, marker=".", label='test error {}'.format(degree))
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title(plot_name)
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("../out/{}".format(plot_name), dpi=512)
    plt.close()
