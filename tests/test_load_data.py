from math import ceil

from src.data import load_datasets


def test_train_split_should_have_70_30_ratio(wine):
    X_train, X_test, y_train, y_test, _, _ = load_datasets(
        test_split=0.3, seed=42
    )

    total_samples = wine.data.shape[0]

    expected_test_samples = ceil(total_samples * 0.30)
    expected_train_samples = total_samples - expected_test_samples

    assert X_train.shape[0] == expected_train_samples
    assert X_test.shape[0] == expected_test_samples

    assert y_train.shape[0] == expected_train_samples
    assert y_test.shape[0] == expected_test_samples


def test_features_should_have_same_number_of_columns_as_original_dataset(wine):
    X_train, X_test, _, _, _, _ = load_datasets(
        test_split=0.3, seed=42
    )

    expected_features = wine.data.shape[1]

    assert X_train.shape[1] == expected_features
    assert X_test.shape[1] == expected_features
