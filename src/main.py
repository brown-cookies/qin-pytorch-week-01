import sys

import torch

from src.config import config
from src.data import load_datasets
from src.train import ManualWineClassification, AutoWineClassificationWithTorch


# Note to myself
# Use Softmax + Cross-Entropy Loss for Multiple classification

# Flow for Multiple classification
# Forward Pass -> Softmax -> Cross-Entropy Loss -> Gradient -> Loop


def main():
    """Entry point of the project."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_test, y_train, y_test, class_names, scaler = load_datasets(
        test_split=config.test_size_split,
        seed=config.seed
    )

    manual_model = ManualWineClassification(
        n_features=X_train.shape[1],
        n_classes=len(class_names),
        cfg=config,
    )

    auto_model = AutoWineClassificationWithTorch()

    manual_loss_per_epoch = manual_model.train_model(X_train, y_train)

    for epoch, loss in enumerate(manual_loss_per_epoch):
        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
