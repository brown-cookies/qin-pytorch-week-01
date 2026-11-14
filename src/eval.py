import sys

import torch
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss
)

from src.train import ManualWineClassification, AutoWineClassificationWithTorch


def evaluate_model(model: ManualWineClassification | AutoWineClassificationWithTorch, X, y, class_names=None):

    if isinstance(model, ManualWineClassification):
        _, _, z = model.forward_pass(X, model.W1, model.b1, model.W2, model.b2)

        probs = model.softmax(z).numpy()
        preds = z.argmax(dim=1).numpy()
        y_true = y.numpy()
        labels = list(range(z.shape[1]))

        return {
            "accuracy": accuracy_score(y_true, preds),
            "loss": log_loss(y_true, probs, labels=labels),
            "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
        }

    model.eval()
    z = model(X)

    probs = torch.softmax(z, dim=1).double().detach().numpy()

    preds = z.argmax(dim=1).numpy()
    y_true = y.numpy()
    labels = list(range(z.shape[1]))

    return {
        "accuracy": accuracy_score(y_true, preds),
        "loss": log_loss(y_true, probs, labels=labels),
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
    }


def compare_model():
    ...
