from methods import matrix_factorization_SGD, ALS
from parsers import load_data, create_submission


def run(lambda_user = 0.001, lambda_item = 0.001, num_features = 30, p_test=0.2):
    train, test = load_data()
    item_features, user_features, rmse = ALS(train, test, lambda_user, lambda_item, num_features)
    # model = NMF(n_components=10, init='random', random_state=0)
    # W = model.fit_transform(train)
    # Z = model.components_
    # print(calculate_mse(test, W.dot(Z)))
    #return rmse
    create_submission(item_features, user_features)


if __name__ == '__main__':
    run()
