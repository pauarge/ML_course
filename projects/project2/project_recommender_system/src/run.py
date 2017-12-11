from helpers import calculate_mse
from methods import matrix_factorization_SGD, ALS, global_mean, compute_std
from parsers import load_data, create_submission


def run(lambda_user = 0.1, lambda_item = 0.01, num_features = 2, min_num_data = 1, p_test=0.2):
    print("LOADING DATA...")
    train, test, transformation_user, transformation_item = load_data(min_num_data)
    print("STARTING MATRIX FACTORIZATION SGD")
    item_features, user_features, rmse = matrix_factorization_SGD(train, test, lambda_user, lambda_item, num_features)
    # model = NMF(n_components=10, init='random', random_state=0)
    # W = model.fit_transform(train)
    # Z = model.components_
    mean = global_mean(train)
    std = compute_std(train)
    create_submission(item_features, user_features, train, transformation_user, transformation_item, mean, std)
    return rmse

if __name__ == '__main__':
    run()
