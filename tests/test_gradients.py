import torch
import torch.nn.functional as F

from src.train import ManualWineClassification


def test_manual_gradient_matches_pytorch_autograd(config):
    torch.manual_seed(42)

    n_samples = 50
    n_features = 13
    n_hidden = 10
    n_classes = 3
    checkpoint_path = "placeholder"

    X = torch.randn((n_samples, n_features))
    y = torch.randint(0, n_classes, (n_samples,))

    manual_model = ManualWineClassification(
        n_features=n_features,
        n_hidden=n_hidden,
        n_classes=n_classes,
        checkpoint_path=checkpoint_path,
        cfg=config
    )

    # Run the manual first
    with torch.no_grad():
        manual_z1, manual_a1, manual_z2 = manual_model.forward_pass(
            X,
            manual_model.W1,
            manual_model.b1,
            manual_model.W2,
            manual_model.b2
        )

        manual_probs = manual_model.softmax(manual_z2)

        manual_loss = manual_model.loss(manual_z2, y)

        manual_dW1, manual_db1, manual_dW2, manual_db2 = manual_model.gradient(
            X,
            manual_z1,
            manual_a1,
            manual_probs,
            y
        )

    auto_W1 = manual_model.W1.clone().requires_grad_(True)
    auto_b1 = manual_model.b1.clone().requires_grad_(True)
    auto_W2 = manual_model.W2.clone().requires_grad_(True)
    auto_b2 = manual_model.b2.clone().requires_grad_(True)

    _, _, auto_z = manual_model.forward_pass(
        X,
        auto_W1,
        auto_b1,
        auto_W2,
        auto_b2
    )

    auto_loss = F.cross_entropy(auto_z, y)

    auto_loss.backward()

    torch.testing.assert_close(
        manual_dW1,
        auto_W1.grad,
    )

    torch.testing.assert_close(
        manual_db1,
        auto_b1.grad,
    )

    torch.testing.assert_close(
        manual_dW2,
        auto_W2.grad,
    )

    torch.testing.assert_close(
        manual_db2,
        auto_b2.grad,
    )
