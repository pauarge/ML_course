from methods import matrix_factorization_SGD, ALS
from parsers import load_data, create_submission


def main():
    train, test = load_data()
    item_features, user_features = ALS(train, test)
    # model = NMF(n_components=10, init='random', random_state=0)
    # W = model.fit_transform(train)
    # Z = model.components_
    # print(calculate_mse(test, W.dot(Z)))
    create_submission(item_features, user_features)


if __name__ == '__main__':
    main()
