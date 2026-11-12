from pathlib import Path

import matplotlib.pyplot as plt

FIGURES_DIR = Path("figures")


def plot_training_history(name, history):
    """Save and show loss curves (train vs val) and validation accuracy."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_loss = history["train_loss_per_epoch"]
    val_loss = history["val_loss_per_epoch"]
    val_acc = history["val_acc_per_epoch"]
    epochs = range(1, len(train_loss) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, train_loss, label="Train")
    ax_loss.plot(epochs, val_loss, label="Validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title(f"{name} loss per epoch")
    ax_loss.legend()
    ax_loss.grid(True)

    ax_acc.plot(epochs, val_acc, color="tab:green")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title(f"{name} validation accuracy")
    ax_acc.grid(True)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / f"{name}_history.png",
        dpi=150
    )
    plt.show()
    plt.close(fig)
