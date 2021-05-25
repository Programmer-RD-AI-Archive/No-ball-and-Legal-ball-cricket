import random
import cv2
import os
import numpy as np
from config import *
from testing.transform_test import *


def load_data_test(imread_type=config["testing"]["imread_type"],img_size=config["testing"]["img_size"]):
    data = []
    index = -1
    labels = {}
    for label in os.listdir("./data/raw/"):
        index += 1
        labels[f"./data/raw/{label}/"] = [index, -1]
    for label in labels:
        for fileimg in os.listdir(label):
            filepath = os.path.join(label, fileimg)
            img = cv2.imread(filepath, imread_type)
            img = cv2.resize(
                img, (img_size,img_size)
            )
            labels[label][1] += 1
            data.append([np.array(transform_test(img)), labels[label][0]])
    for _ in range(random.randint(1, 100)):
        np.random.shuffle(data)
    np.save("./data/cleaned/data.npy", data)
    X = []
    y = []
    for d in data:
        X.append(d[0])
        y.append(d[1])
    VAL_SPLIT = 25
    X_train = torch.from_numpy(np.array(X[:-VAL_SPLIT]))
    y_train = torch.from_numpy(np.array(y[:-VAL_SPLIT]))
    X_test = torch.from_numpy(np.array(X[-VAL_SPLIT:]))
    y_test = torch.from_numpy(np.array(y[-VAL_SPLIT:]))
    print(len(X_train))
    print(len(X_test))
    np.save("./data/cleaned/X.npy", np.array(X))
    np.save("./data/cleaned/y.npy", np.array(y))
    np.save("./data/cleaned/X_train.npy", np.array(X_train))
    np.save("./data/cleaned/X_test.npy", np.array(X_test))
    np.save("./data/cleaned/y_train.npy", np.array(y_train))
    np.save("./data/cleaned/y_test.npy", np.array(y_test))
    return data, X, y, X_train, X_test, y_train, y_test
