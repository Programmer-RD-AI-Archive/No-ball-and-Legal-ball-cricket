import wandb
from tqdm import tqdm
from log import *
from config import *
import torch


def loss(criterion, y, model, X):
    print("model to cpu")
    model.to("cpu")
    print("predicting")
    preds = model(
        X.view(
            -1,
            config["testing"]["color_channel(s)"],
            config["testing"]["img_size"],
            config["testing"]["img_size"],
        )
        .to("cpu")
        .float()
    )
    print("preds to cpu")
    preds.to("cpu")
    print("creating loss")
    loss = criterion(preds, y).to("cpu")
    print("loss.backward()")
    loss.backward()
    print("returning")
    return loss.item()


def accuracy(net, X, y):
    device = "cpu"
    net.to(device)
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for i in range(len(X)):
            real_class = torch.argmax(y[i]).to(device)
            net_out = net(
                X[i].to(device).float()
            )
            net_out = net_out[0]
            predictied_class = torch.argmax(net_out)
            if predictied_class == real_class:
                correct += 1
            total += 1
    net.train()
    return round(correct / total, 3)


def train(
    BATCH_SIZE,
    EPOCHS,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    device,
    optimizer,
    criterion,
    name,
    PROJECT_NAME,
):
    wandb.init(project=PROJECT_NAME, name=name)
    device = torch.device(device)
    for _ in tqdm(range(EPOCHS), leave=False):
        for i in range(0, len(X_train), BATCH_SIZE):
            X_batch = X_train[i : i + BATCH_SIZE].to(device)
            y_batch = y_train[i : i + BATCH_SIZE].to(device)
            model.to(device)
            preds = model(X_batch)
            preds.to(device)
            loss = criterion(preds, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if f"{name}-per-iter.csv" in os.listdir("./output/logs/"):
                log(
                    f"{name}-per-iter",
                    {
                        "loss": loss.item(),
                        "accuracy": accuracy(model, X_train, y_train),
                        "val_accuracy": accuracy(model, X_test, y_test),
                    },
                )
            else:
                log(
                    f"{name}-per-iter",
                    {
                        "loss": [loss.item()],
                        "accuracy": [accuracy(model, X_train, y_train)],
                        "val_accuracy": [accuracy(model, X_test, y_test)],
                    },
                )
        wandb.log(
            {
                "loss": loss.item(),
                "accuracy": accuracy(model, X_train, y_train),
                "val_accuracy": accuracy(model, X_test, y_test),
            }
        )
        if f"{name}-per-epoch.csv" in os.listdir("./output/logs/"):
            log(
                f"{name}-per-epoch",
                {
                    "loss": loss.item(),
                    "accuracy": accuracy(model, X_train, y_train),
                    "val_accuracy": accuracy(model, X_test, y_test),
                },
            )
        else:
            log(
                f"{name}-per-epoch",
                {
                    "loss": [loss.item()],
                    "accuracy": [accuracy(model, X_train, y_train)],
                    "val_accuracy": [accuracy(model, X_test, y_test)],
                },
            )
    return model, y_batch, preds
