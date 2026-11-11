import matplotlib.pyplot as plt

path = "figures"


def plot_loss_per_epoch(name, loss_per_epoch):
    plt.figure()
    plt.plot(range(1, len(loss_per_epoch) + 1), loss_per_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"{name} loss per epoch")
    plt.grid(True)
    plt.savefig(f"{path}/{name}_loss.png")
    plt.show()
