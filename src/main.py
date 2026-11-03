import sys

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


RANDOM_STATE = 42
TEST_SIZE_SPLIT = 0.3

EPOCH = 10
LEARNING_RATE = 0.01


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

    class_names = [str(name) for name in wine.target_names]

    return X_train, X_test, y_train, y_test, class_names, scaler


def init_weights_biases(n_features, n_classes, seed=RANDOM_STATE):
    """Create the starting weights and bias for one linear layer."""
    generator = torch.Generator().manual_seed(seed)

    W = torch.randn(n_features, n_classes, generator=generator)
    b = torch.zeros(n_classes)

    return W, b


def forward_pass(X, W, b):
    """One linear layer manual forward pass"""
    z = X @ W + b
    return z


def loss_function(pred, actual):

    return ...


def main():
    """Entry point of the project."""

    X_train, X_test, y_train, y_test, class_names, scaler = load_data()

    n_features = X_train.data.shape[1]
    n_classes = len(class_names)

    W, b = init_weights_biases(n_features, n_classes)
    z = forward_pass(X_train, W, b)

    return 0


if __name__ == "__main__":
    sys.exit(main())
