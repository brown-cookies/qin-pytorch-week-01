import numpy as np

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch


def split_indices(*, y, val_split: float, test_split: float, seed: int):

    indices = np.arange(len(y))

    train_val_index, test_index = train_test_split(
        indices,
        test_size=test_split,
        stratify=y,
        random_state=seed,
    )

    val_fraction = val_split / (1 - test_split)

    train_index, val_index = train_test_split(
        train_val_index,
        test_size=val_fraction,
        stratify=y[train_val_index],
        random_state=seed,
    )

    return {
        "train": train_index,
        "validation": val_index,
        "test": test_index,
    }


def load_datasets(*, val_split: float, test_split: float, seed: int):
    """
    Load the Wine dataset, split it, standardize the features,
    and convert everything into PyTorch tensors.
    """
    wine = load_wine()

    X = wine.data
    y = wine.target

    split = split_indices(
        y=y,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )

    X_train = X[split["train"]]
    X_val = X[split["validation"]]
    X_test = X[split["test"]]

    y_train = y[split["train"]]
    y_val = y[split["validation"]]
    y_test = y[split["test"]]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    class_names = [str(name) for name in wine.target_names]

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "class_names": class_names,
        "scaler": scaler
    }
