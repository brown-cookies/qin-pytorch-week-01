import sys

import torch

from src.config import config
from src.data import load_datasets
from src.eval import evaluate_model, compare_model
from src.train import ManualWineClassification, AutoWineClassificationWithTorch


# Note to myself
# Use Softmax + Cross-Entropy Loss for Multiple classification

# Flow for Multiple classification
# Forward Pass -> Softmax -> Cross-Entropy Loss -> Gradient -> Loop

# TODO: Add print summary
def _print_training_summary():
    """
    Prints the training summary of either Manual or Auto
    with their respective evaluation
    """

    ...


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
        checkpoint_path="runs/manual/",
        cfg=config,
    )

    # auto_model = AutoWineClassificationWithTorch()

    manual_loss_per_epoch = manual_model.train_model(
        X_train, y_train, is_resume=False
    )

    manual_model.load_checkpoint("best_run.pt")

    result = evaluate_model(manual_model, X_test, y_test,
                            class_names=list(class_names))

    print(f"loss:     {result['loss']:.4f}")
    print(f"accuracy: {result['accuracy']:.2%}")
    print(f"macro f1: {result['macro_f1']:.2%}")
    print(result["confusion"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
