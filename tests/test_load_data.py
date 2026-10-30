from math import ceil

from src.main import load_data


def test_train_split_should_have_70_30_ratio(wine):
    X_train, X_test, y_train, y_test, scaler = load_data()

    total_samples = wine.data.shape[0]

    expected_test_samples = ceil(total_samples * 0.30)
    expected_train_samples = total_samples - expected_test_samples

    assert X_train.shape[0] == expected_train_samples
    assert X_test.shape[0] == expected_test_samples

    assert y_train.shape[0] == expected_train_samples
    assert y_test.shape[0] == expected_test_samples


def test_features_should_have_same_number_of_columns_as_original_dataset(wine):
    X_train, X_test, _, _, _ = load_data()

    expected_features = wine.data.shape[1]

    assert X_train.shape[1] == expected_features
    assert X_test.shape[1] == expected_features
