import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_a_file(data, name):
    plt.figure(figsize=(12, 7))
    plt.show(data)
    plt.savefig(f"./output/plots/{name}.png")
    plt.show()


def plot(data, name, per_iter=False, per_epochs=True):
    for col in list(data.columns):
        if per_epochs:
            plot_a_file(col, f"{name}-per-iter-{col}")
        else:
            plot_a_file(col, f"{name}-per-epoch-{col}")
    return True
