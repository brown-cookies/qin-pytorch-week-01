import torch
import torch.nn as nn


from src.config import Config


class ManualWineClassification():
    def __init__(self, n_features, n_classes, cfg: Config):
        self.cfg = cfg
        self.W, self.b = self.init_weights_biases(
            n_features, n_classes, self.cfg.seed
        )

    def init_weights_biases(self, n_features, n_classes, seed: int):
        """Create the starting weights and bias for one linear layer."""
        generator = torch.Generator().manual_seed(seed)

        W = torch.randn(n_features, n_classes, generator=generator)
        b = torch.zeros(n_classes)

        return W, b

    def forward_pass(self, X, W, b):
        """One linear layer manual forward pass"""
        z = X @ W + b

        return z

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

    def gradient(self, X, pred, actual):
        """Manual hand-computed gradient for Softmax/Cross-Entropy Loss"""

        n_samples = X.shape[0]

        dz = pred.clone()

        rows = torch.arange(n_samples)

        dz[rows, actual] -= 1

        dz /= n_samples

        dW = X.T @ dz

        db = dz.sum(dim=0)

        return dW, db

    def train_model(self, X_train, y_train):
        epochs = self.cfg.epochs
        learning_rate = self.cfg.learning_rate

        loss_per_epoch = []

        for epoch in range(epochs):
            z = self.forward_pass(X_train, self.W, self.b)

            pred = self.softmax(z)

            loss = self.loss(z, y_train)

            loss_per_epoch.append(loss.item())

            dW, db = self.gradient(X_train, pred, y_train)

            with torch.no_grad():
                self.W = self.W - learning_rate * dW
                self.b = self.b - learning_rate * db

        return loss_per_epoch


class AutoWineClassificationWithTorch(nn.Module):
    def __init__(self, n_features, n_classes, cfg: Config):
        self.cfg = cfg
        self.linear = nn.Linear(n_features, n_classes)

    @staticmethod
    def forward(self, X, W, b):
        return self.linear(X, W, b)
