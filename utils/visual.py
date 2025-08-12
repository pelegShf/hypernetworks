import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def plot_results(history, save_prefix=None):
    epochs = range(1, len(history["test_acc"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["test_loss"], label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_loss.png", bbox_inches="tight", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["test_acc"], label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")

    if save_prefix:
        plt.savefig(f"{save_prefix}_acc.png", bbox_inches="tight", dpi=150)
    plt.close()
