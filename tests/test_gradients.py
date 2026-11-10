import torch
import torch.nn.functional as F

from src.train import ManualWineClassification


def test_manual_gradient_matches_pytorch_autograd(config):
    torch.manual_seed(42)

    n_samples = 50
    n_features = 13
    n_classes = 3
    checkpoint_path = "placeholder"

    X = torch.randn((n_samples, n_features))
    y = torch.randint(0, n_classes, (n_samples,))

    manual_model = ManualWineClassification(
        n_features=n_features,
        n_classes=n_classes,
        checkpoint_path=checkpoint_path,
        cfg=config
    )

    # Run the manual first
    with torch.no_grad():
        manual_z = manual_model.forward_pass(X, manual_model.W, manual_model.b)

        manual_probs = manual_model.softmax(manual_z)

        manual_loss = manual_model.loss(manual_z, y)

        manual_dW, manual_db = manual_model.gradient(X, manual_probs, y)

    auto_W = manual_model.W.clone().requires_grad_(True)
    auto_b = manual_model.b.clone().requires_grad_(True)

    auto_z = manual_model.forward_pass(X, auto_W, auto_b)

    auto_loss = F.cross_entropy(auto_z, y)

    auto_loss.backward()

    torch.set_printoptions(precision=10)

    print("\nManual")
    print(manual_db)
    print("Auto")
    print(auto_b.grad)

    torch.testing.assert_close(
        manual_dW,
        auto_W.grad,
    )

    torch.testing.assert_close(
        manual_db,
        auto_b.grad,
    )
