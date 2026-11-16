import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

from sklearn.datasets import load_wine

from src.cli import parse_args
from src.config import config
from src.data import load_datasets, split_indices
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


def run(model="both"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_datasets(
        val_split=config.val_size_split,
        test_split=config.test_size_split,
        seed=config.seed
    )

    run_manual = model in ("both", "manual")
    run_auto = model in ("both", "auto")

    if run_manual:
        manual_model = ManualWineClassification(
            n_features=data["X_train"].shape[1],
            n_hidden=20,
            n_classes=len(data["class_names"]),
            checkpoint_path="runs/manual/",
            cfg=config,
        )

        manual_history = manual_model.train_model(
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            is_resume=False
        )

        manual_checkpoint = manual_model.load_checkpoint("best_run.pt")

        manual_result = evaluate_model(
            manual_model,
            data["X_test"],
            data["y_test"],
            class_names=list(data["class_names"])
        )

        print("[manual]")
        print(f"loss:     {manual_result['loss']:.4f}")
        print(f"accuracy: {manual_result['accuracy']:.2%}")
        print(f"macro f1: {manual_result['macro_f1']:.2%}")

    if run_auto:
        auto_model = AutoWineClassificationWithTorch(
            n_features=data["X_train"].shape[1],
            n_hidden=20,
            n_classes=len(data["class_names"]),
            checkpoint_path="runs/auto/",
            cfg=config,
        )

        auto_history = auto_model.train_model(
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            is_resume=False
        )

        auto_checkpoint = auto_model.load_checkpoint("best_run.pt")

        auto_result = evaluate_model(
            auto_model,
            data["X_test"],
            data["y_test"],
            class_names=list(data["class_names"])
        )

        print("[auto]")
        print(f"loss:     {auto_result['loss']:.4f}")
        print(f"accuracy: {auto_result['accuracy']:.2%}")
        print(f"macro f1: {auto_result['macro_f1']:.2%}")

    if run_manual:
        plot_training_history("manual", manual_history)

    if run_auto:
        plot_training_history("auto", auto_history)


def train(out, is_smoke=False):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if is_smoke:
        rows = 32
        cfg = replace(config, epochs=20)
    else:
        cfg = config

    data = load_datasets(
        val_split=cfg.val_size_split,
        test_split=cfg.test_size_split,
        seed=cfg.seed
    )

    X_train = data["X_train"]
    y_train = data["y_train"]

    if is_smoke:
        X_train = X_train[:rows]
        y_train = y_train[:rows]

    n_features = X_train.shape[1]
    n_hidden = 20
    n_classes = len(data["class_names"])

    model = ManualWineClassification(
        n_features=n_features,
        n_hidden=n_hidden,
        n_classes=n_classes,
        checkpoint_path=str(out.parent / f"{out.stem}"),
        cfg=cfg,
    )

    history = model.train_model(
        X_train,
        y_train,
        data["X_val"],
        data["y_val"],
        is_resume=False,
    )


def evaluate(checkpoint, out):
    checkpoint_path = Path(checkpoint)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_data = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # Load the automatic model
    if "model" in checkpoint_data:
        state_dict = checkpoint_data["model"]
        W1 = state_dict["forward_pass.0.weight"]
        W2 = state_dict["forward_pass.2.weight"]

        n_hidden, n_features = W1.shape
        n_classes, _ = W2.shape

        model = AutoWineClassificationWithTorch(
            n_features=n_features,
            n_hidden=n_hidden,
            n_classes=n_classes,
            checkpoint_path=str(checkpoint_path.parent),
            cfg=config,
        )

        model.load_state_dict(state_dict)

        model.eval()

    # Defaults to manual
    else:
        W1 = checkpoint_data["W1"]
        b1 = checkpoint_data["b1"]
        W2 = checkpoint_data["W2"]
        b2 = checkpoint_data["b2"]

        n_features, n_hidden = W1.shape
        _, n_classes = W2.shape

        model = ManualWineClassification(
            n_features=n_features,
            n_hidden=n_hidden,
            n_classes=n_classes,
            checkpoint_path=str(checkpoint_path.parent),
            cfg=config,
        )

        model.W1 = W1
        model.b1 = b1
        model.W2 = W2
        model.b2 = b2

    data = load_datasets(
        val_split=config.val_size_split,
        test_split=config.test_size_split,
        seed=config.seed,
    )

    result = evaluate_model(
        model,
        data["X_test"],
        data["y_test"],
        class_names=list(data["class_names"]),
    )

    metrics = {
        "test_accuracy": float(result["accuracy"]),
        "test_loss": float(result["loss"]),
        "test_macro_f1": float(result["macro_f1"]),
    }

    out.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )


def split(out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    wine = load_wine()

    y = wine.target

    split = split_indices(
        y=y,
        val_split=config.val_size_split,
        test_split=config.test_size_split,
        seed=config.seed,
    )

    split_data = {
        key: value.tolist()
        for key, value in split.items()
    }

    out.write_text(
        json.dumps(split_data, indent=4),
        encoding="utf-8"
    )


def main(argv=None):
    """Entry point of the project."""

    args = parse_args(argv)

    if args.command == "run":
        run(args.model)
    elif args.command == "train":
        train(args.out, args.smoke)
    elif args.command == "evaluate":
        evaluate(args.checkpoint, args.metrics_out)
    elif args.command == "split":
        split(args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
