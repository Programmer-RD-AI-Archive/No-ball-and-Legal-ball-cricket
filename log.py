import pandas as pd
import os


def log(name="test", info={"-": "-"}):
    if f"{name}.csv" in os.listdir("./output/logs/"):
        data = pd.read_csv(f"./output/logs/{name}.csv")
        data = data.append(info
        , ignore_index=True)
    else:
        data = pd.DataFrame(info)
    data.to_csv(f"./output/logs/{name}.csv", index=False)
