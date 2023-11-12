import os
import csv
import dataclasses
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


@dataclasses.dataclass
class Params:
    q: float
    alpha: float
    rounds: int
    local_update:  int
    batch_size: int
    num_clients: int
    sampling_type: str

def parse_filename(filename:str) -> Params:
    tokens = filename.replace(".txt", "").split("+")
    params = Params(
        uni_sampling_size=int(tokens[0].split(":")[0]),
        arb_sampling_size=int(tokens[0].split(":")[1]),
        q =float(tokens[1].replace('q_', '')),
        bias=float(tokens[2].replace('bias', '')),
        epoch=int(tokens[3].replace('epoch', '')),
        local_update=int(tokens[4].replace('local', '')),
        batch_size=int(tokens[5].replace('batch', '')),
        n_workers =int(tokens[6].replace('nwork', '')),
        sample_type=tokens[7].replace('sampling_', '')
    )
    return params


df = pd.read_csv("results/uniform.csv")
df.plot(x="Step", y="Value")

plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.grid("on")
plt.show()

