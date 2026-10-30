import sys

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


RANDOM_STATE = 42
TEST_SIZE_SPLIT = 0.3


def load_data():
    """
    Load the Wine dataset, split it, standardize the features,
    and convert everything into PyTorch tensors.
    """

    wine = load_wine()

    X = wine.data
    y = wine.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE_SPLIT,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test, scaler


def main():
    """Entry point of the project."""

    X_train, X_test, y_train, y_test, scaler = load_data()

    return 0


if __name__ == "__main__":
    sys.exit(main())
