from math import ceil

from src.data import load_datasets


def test_train_split_should_have_70_15_15_ratio(wine):
    d = load_datasets(
        val_split=0.15, test_split=0.15, seed=42
    )

    X_train = d["X_train"]
    X_val = d["X_val"]
    X_test = d["X_test"]
    y_train = d["y_train"]
    y_val = d["y_val"]
    y_test = d["y_test"]

    total_samples = wine.data.shape[0]

    expected_test_samples = ceil(total_samples * 0.15)
    expected_val_samples = ceil(total_samples * 0.15)
    expected_train_samples = total_samples - \
        expected_val_samples - expected_test_samples

    assert X_train.shape[0] == expected_train_samples
    assert X_val.shape[0] == expected_val_samples
    assert X_test.shape[0] == expected_test_samples

    assert y_train.shape[0] == expected_train_samples
    assert y_val.shape[0] == expected_val_samples
    assert y_test.shape[0] == expected_test_samples


def test_features_should_have_same_number_of_columns_as_original_dataset(wine):
    d = load_datasets(
        val_split=0.15, test_split=0.15, seed=42
    )

    X_train = d["X_train"]
    X_test = d["X_test"]

    expected_features = wine.data.shape[1]

    assert X_train.shape[1] == expected_features
    assert X_test.shape[1] == expected_features
