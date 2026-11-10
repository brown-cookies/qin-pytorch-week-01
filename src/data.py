from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch


def load_datasets(*, test_split: int, seed: int):
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
        test_size=test_split,
        stratify=y,
        random_state=seed,
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
