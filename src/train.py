from pathlib import Path

import torch
import torch.nn as nn


from src.config import Config


class ManualWineClassification():
    def __init__(self, n_features, n_hidden, n_classes, checkpoint_path, cfg: Config):
        self.checkpoint_path = checkpoint_path
        self.cfg = cfg
        self.W1, self.b1, self.W2, self.b2 = self.init_weights_biases(
            n_features, n_hidden, n_classes, self.cfg.seed
        )

    def init_weights_biases(self, n_features, n_hidden, n_classes, seed: int):
        """Create the starting weights and bias for two linear layer."""
        generator = torch.Generator().manual_seed(seed)

        W1 = torch.randn(n_features, n_hidden, generator=generator)
        b1 = torch.zeros(n_hidden)

        W2 = torch.randn(n_hidden, n_classes, generator=generator)
        b2 = torch.zeros(n_classes)

        return W1, b1, W2, b2

    def forward_pass(self, X, W1, b1, W2, b2):
        """Two linear layer manual forward pass"""
        z1 = X @ W1 + b1
        a1 = torch.relu(z1)
        z2 = a1 @ W2 + b2
        return z1, a1, z2

    def softmax(self, pred):
        """Convert raw-score into probabilities"""
        shifted = pred - pred.max(dim=1, keepdim=True).values

        exp_values = torch.exp(shifted)

        prob = exp_values / exp_values.sum(
            dim=1, keepdim=True
        )

        return prob

    def loss(self, pred, actual):
        """Manual Cross-entropy loss for Multiple classification model"""
        n_samples = pred.shape[0]

        shifted = pred - pred.max(dim=1, keepdim=True).values

        log_sum_exp = torch.log(
            torch.exp(shifted).sum(dim=1)
        )

        rows = torch.arange(n_samples)

        correct_class_logits = shifted[rows, actual]

        loss = (
            log_sum_exp - correct_class_logits
        ).mean()

        return loss

    def gradient(self, X, z1, a1, pred, actual):
        """Manual hand-computed gradient for Softmax/Cross-Entropy Loss"""

        n_samples = X.shape[0]
        rows = torch.arange(n_samples)

        dz2 = pred.clone()

        dz2[rows, actual] -= 1

        dz2 /= n_samples

        dW2 = a1.T @ dz2

        db2 = dz2.sum(dim=0)

        da1 = dz2 @ self.W2.T

        dz1 = da1 * (z1 > 0)

        dW1 = X.T @ dz1

        db1 = dz1.sum(dim=0)

        return dW1, db1, dW2, db2

    def save_checkpoint(self, name, epoch, train_loss_per_epoch, val_acc_per_epoch, val_loss_per_epoch, best_val_loss):
        """Save a training checkpoint"""
        directory = Path(self.checkpoint_path)
        directory.mkdir(parents=True, exist_ok=True)

        path = directory / name

        state = {
            "epoch": epoch,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "train_loss_per_epoch": train_loss_per_epoch,
            "val_acc_per_epoch": val_acc_per_epoch,
            "val_loss_per_epoch": val_loss_per_epoch,
            "best_val_loss": best_val_loss,
        }

        torch.save(state, path)

    def load_checkpoint(self, name):
        """Load a training checkpoint"""
        path = Path(self.checkpoint_path) / name
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")

        self.W1 = checkpoint["W1"]
        self.b1 = checkpoint["b1"]
        self.W2 = checkpoint["W2"]
        self.b2 = checkpoint["b2"]

        return checkpoint

    def evaluate_val(self, X, y):
        """"""
        _, _, z2 = self.forward_pass(X, self.W1, self.b1, self.W2, self.b2)

        loss = self.loss(z2, y)

        pred = z2.argmax(dim=1)

        acc = (pred == y).float().mean().item()

        return acc, loss.item()

    def train_model(self, X_train, y_train, X_val, y_val, is_resume=False):
        epochs = self.cfg.epochs
        learning_rate = self.cfg.learning_rate

        best_val_loss = float("inf")

        train_loss_per_epoch = []
        val_acc_per_epoch = []
        val_loss_per_epoch = []

        start = 0

        if is_resume:
            checkpoint = self.load_checkpoint(
                "last_run.pt"
            )

            last_epoch = checkpoint["epoch"]
            best_val_loss = checkpoint["best_val_loss"]
            train_loss_per_epoch = checkpoint["train_loss_per_epoch"]
            val_acc_per_epoch = checkpoint["val_acc_per_epoch"]
            val_loss_per_epoch = checkpoint["val_loss_per_epoch"]

            start = last_epoch + 1

        for epoch in range(start, epochs):
            z1, a1, z2 = self.forward_pass(
                X_train, self.W1, self.b1, self.W2, self.b2
            )

            pred = self.softmax(z2)

            dW1, db1, dW2, db2 = self.gradient(X_train, z1, a1, pred, y_train)

            with torch.no_grad():
                self.W1 = self.W1 - learning_rate * dW1
                self.b1 = self.b1 - learning_rate * db1
                self.W2 = self.W2 - learning_rate * dW2
                self.b2 = self.b2 - learning_rate * db2

            train_loss = self.loss(z2, y_train)

            val_acc, val_loss = self.evaluate_val(
                X_val,
                y_val
            )

            train_loss_per_epoch.append(train_loss.item())
            val_acc_per_epoch.append(val_acc)
            val_loss_per_epoch.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(
                    "best_run.pt",
                    epoch,
                    train_loss_per_epoch,
                    val_acc_per_epoch,
                    val_loss_per_epoch,
                    best_val_loss
                )

        self.save_checkpoint(
            "last_run.pt",
            epoch,
            train_loss_per_epoch,
            val_acc_per_epoch,
            val_loss_per_epoch,
            best_val_loss
        )

        return {
            "train_loss_per_epoch": train_loss_per_epoch,
            "val_acc_per_epoch": val_acc_per_epoch,
            "val_loss_per_epoch": val_loss_per_epoch
        }


class AutoWineClassificationWithTorch(nn.Module):
    def __init__(self, n_features, n_classes, cfg: Config):
        super().__init__()

        self.cfg = cfg
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, X):
        return self.linear(X)
