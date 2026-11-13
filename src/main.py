import sys

import torch

from src.config import config
from src.data import load_datasets
from src.eval import evaluate_model, compare_model
from src.plot import plot_training_history
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

    data = load_datasets(
        val_split=config.val_size_split,
        test_split=config.test_size_split,
        seed=config.seed
    )

    manual_model = ManualWineClassification(
        n_features=data["X_train"].shape[1],
        n_hidden=20,
        n_classes=len(data["class_names"]),
        checkpoint_path="runs/manual/",
        cfg=config,
    )

    auto_model = AutoWineClassificationWithTorch(
        n_features=data["X_train"].shape[1],
        n_hidden=20,
        n_classes=len(data["class_names"]),
        checkpoint_path="runs/auto/",
        cfg=config,
    )

    manual_history = manual_model.train_model(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        is_resume=False
    )

    auto_history = auto_model.train_model(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        is_resume=False
    )

    manual_checkpoint = manual_model.load_checkpoint("best_run.pt")
    auto_checkpoint = auto_model.load_checkpoint("best_run.pt")

    manual_result = evaluate_model(
        manual_model,
        data["X_test"],
        data["y_test"],
        class_names=list(data["class_names"])
    )

    auto_result = evaluate_model(
        auto_model,
        data["X_test"],
        data["y_test"],
        class_names=list(data["class_names"])
    )

    print(f"loss:     {manual_result['loss']:.4f}")
    print(f"accuracy: {manual_result['accuracy']:.2%}")
    print(f"macro f1: {manual_result['macro_f1']:.2%}")

    print(f"loss:     {auto_result['loss']:.4f}")
    print(f"accuracy: {auto_result['accuracy']:.2%}")
    print(f"macro f1: {auto_result['macro_f1']:.2%}")

    plot_training_history("manual", manual_history)
    plot_training_history("auto", auto_history)

    return 0


if __name__ == "__main__":
    sys.exit(main())
