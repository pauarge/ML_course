from sklearn.decomposition import NMF

from helpers import split_data, calculate_mse, plot_raw_data
from parsers import load_data

NUM_ITEMS_PER_USER = 1
NUM_USERS_PER_ITEM = 1
MIN_NUM_RATINGS = 10


def main():
    data = load_data()
    num_items_per_user, num_users_per_item = plot_raw_data(data)
    valid_data, train, test = split_data(data, NUM_ITEMS_PER_USER, NUM_USERS_PER_ITEM, MIN_NUM_RATINGS,
                                         p_test=0.1)
    model = NMF(n_components=10, init='random', random_state=0)
    W = model.fit_transform(train)
    Z = model.components_
    print(calculate_mse(test, W.dot(Z)))


if __name__ == '__main__':
    main()
