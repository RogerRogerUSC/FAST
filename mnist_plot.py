import os
import csv
import dataclasses
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


@dataclasses.dataclass
class Params:
    sampling_type: str
    q: float
    alpha: float
    num_clients: int


def parse_filename(filename: str) -> Params:
    tokens = filename.replace(".csv", "").split("+")
    params = Params(
        sampling_type=tokens[0].replace("run-", ""),
        q=float(tokens[1].replace("q_", "")),
        alpha=float(tokens[2].replace("alpha_", "")),
        num_clients=int(
            tokens[3].replace("num_clients_", "").replace("-tag-Accuracy_test", "")
        ),
    )
    return params


fig, ax = plt.subplots(figsize=(12, 8))
markers = "os*xdXDHhPp12348<>"
k = 0

filenames = os.listdir("results/")

for filename in filenames: 
    params = parse_filename(filename)
    # Select sampling_type to plot
    if "cyclic" in params.sampling_type or params.sampling_type=="uniform": 
        df = pd.read_csv("results/" + filename)
        if "_" in params.sampling_type:
            df.plot(
            x="Step",
            y="Value",
                label=params.sampling_type + ",q=" + str(params.q),
                ax=ax,
                marker=markers[k],
                linewidth="1.1",
                markersize=5,
                markevery=2,
            )
        else:
            df.plot(
                x="Step",
                y="Value",
                label=params.sampling_type,
                ax=ax,
                marker=markers[k],
                linewidth="1.1",
                markersize=5,
                markevery=2,
            )
        k += 1

plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.title("dataset=MNIST, alpha=0.1, num_clients=100, learning_rate=0.01, rounds=500")
plt.grid("on")
plt.show()
plt.savefig(os.path.join("figures", "cyclic_multi_q.pdf"))
