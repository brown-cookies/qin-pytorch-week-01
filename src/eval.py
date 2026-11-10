import sys

import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, log_loss
)

# TODO: Add model evaluation, both for manual and auto
# TODO: Add comparison evaluation between manual and auto


def evaluate_model(model, X, y, class_names=None):
    z = model.forward_pass(X, model.W, model.b)

    probs = model.softmax(z).numpy()
    preds = z.argmax(dim=1).numpy()
    y_true = y.numpy()
    labels = list(range(z.shape[1]))

    return {
        "accuracy": accuracy_score(y_true, preds),
        "confusion": confusion_matrix(y_true, preds, labels=labels),
        "loss": log_loss(y_true, probs, labels=labels),
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
    }


def compare_model():
    ...
