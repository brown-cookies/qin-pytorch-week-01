import torch

from src.train import ManualWineClassification, AutoWineClassificationWithTorch

# what needs to be identical
# the n_features, n_hidden, n_classes
# the weight and biases
# the config setup, learning_rate, epochs, seed


def test_manual_and_auto_agrees_with_each_other(config):
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

    auto_model = AutoWineClassificationWithTorch(
        n_features=n_features,
        n_hidden=n_hidden,
        n_classes=n_classes,
        checkpoint_path=checkpoint_path,
        cfg=config
    )

    optimizer = torch.optim.SGD(
        auto_model.parameters(),
        lr=config.learning_rate
    )

    # Sync first the weight and biases for the comparison to mean anything
    with torch.no_grad():
        auto_model.forward_pass[0].weight.copy_(manual_model.W1.t())
        auto_model.forward_pass[0].bias.copy_(manual_model.b1)
        auto_model.forward_pass[2].weight.copy_(manual_model.W2.t())
        auto_model.forward_pass[2].bias.copy_(manual_model.b2)

    _, _, manual_z = manual_model.forward_pass(
        X,
        manual_model.W1,
        manual_model.b1,
        manual_model.W2,
        manual_model.b2
    )

    auto_z = auto_model(X)

    torch.testing.assert_close(manual_z, auto_z)

    manual_loss = manual_model.loss(manual_z, y)

    auto_loss = auto_model.loss(auto_z, y)

    torch.testing.assert_close(manual_loss, auto_loss)

    for epoch in range(5):
        z1, a1, z2 = manual_model.forward_pass(
            X,
            manual_model.W1,
            manual_model.b1,
            manual_model.W2,
            manual_model.b2
        )
        pred = manual_model.softmax(z2)
        dW1, db1, dW2, db2 = manual_model.gradient(X, z1, a1, pred, y)

        with torch.no_grad():
            manual_model.W1 -= config.learning_rate * dW1
            manual_model.b1 -= config.learning_rate * db1
            manual_model.W2 -= config.learning_rate * dW2
            manual_model.b2 -= config.learning_rate * db2

        auto_model.train()
        auto_model.zero_grad()
        z = auto_model(X)
        train_loss = auto_model.loss(z, y)
        train_loss.backward()
        optimizer.step()

        torch.testing.assert_close(
            dW1, auto_model.forward_pass[0].weight.grad.t())
        torch.testing.assert_close(db1, auto_model.forward_pass[0].bias.grad)
        torch.testing.assert_close(
            dW2, auto_model.forward_pass[2].weight.grad.t())
        torch.testing.assert_close(db2, auto_model.forward_pass[2].bias.grad)

    _, _, manual_z = manual_model.forward_pass(
        X,
        manual_model.W1,
        manual_model.b1,
        manual_model.W2,
        manual_model.b2
    )

    auto_z = auto_model(X)

    torch.testing.assert_close(manual_z, auto_z)

    manual_loss = manual_model.loss(manual_z, y)

    auto_loss = auto_model.loss(auto_z, y)

    torch.testing.assert_close(manual_loss, auto_loss)
