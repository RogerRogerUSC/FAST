import os
import csv
import dataclasses
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import argparse
from scipy.interpolate import make_interp_spline, BSpline

parser = argparse.ArgumentParser(description="Fashion-MNIST Plot")
parser.add_argument(
    "--sampling-type",
    type=str,
    default="beta",
)
parser.add_argument("--alpha", type=float, default=0.05)
args = parser.parse_args()


@dataclasses.dataclass
class Params:
    train_batch_size: int
    test_batch_size: int
    lr: float
    sampling_type: str
    local_update: int
    num_clients: int
    rounds: int
    q: float
    alpha: float
    data_dist: str


def parse_filename(filename: str) -> Params:
    tokens = filename.replace(".csv", "").split(",")
    params = Params(
        train_batch_size=int(tokens[0].replace("run-Namespace(train_batch_size=", "")),
        test_batch_size=int(tokens[1].replace("test_batch_size=", "")),
        lr=float(tokens[2].replace("lr=", "")),
        sampling_type=tokens[3].replace("sampling_type='", "").replace("'", ""),
        local_update=int(tokens[4].replace("local_update=", "")),
        num_clients=int(tokens[5].replace("num_clients=", "")),
        rounds=int(tokens[6].replace("rounds=", "")),
        q=float(tokens[7].replace("q=", "")),
        alpha=float(tokens[8].replace("alpha=", "")),
        data_dist=tokens[9].replace("data_dist='", "").replace("'", ""),
    )
    return params


fig, ax = plt.subplots(figsize=(6, 5))
plt.rc("font", size=12)
markers = "os*xdXDHhPp12348<>"
k = 0
filenames = os.listdir(f"results/fashion/{args.sampling_type}_alpha_{args.alpha}/")


for filename in filenames:
    params = parse_filename(filename)
    print(params.sampling_type)
    # Select sampling_type to plot
    if args.sampling_type in params.sampling_type or params.sampling_type == "uniform":
        if params.q != "0.7":
            df = pd.read_csv(f"results/fashion/{args.sampling_type}_alpha_{args.alpha}/{filename}")
            xnew = np.linspace(df["Step"].min(), df["Step"].max(), 200) 
            spl = make_interp_spline(df["Step"], df["Value"], k=5)
            y_smooth = BSpline(xnew)
            if "_" in params.sampling_type:
                ax.plot(
                    xnew,
                    y_smooth,
                    label=params.sampling_type + ",q=" + str(params.q),
                    marker=markers[k],
                    linewidth="1.1",
                    markersize=3,
                    markevery=2,
                )
            else:
                ax.plot(
                    xnew,
                    y_smooth,
                    label=params.sampling_type,
                    marker=markers[k],
                    linewidth="1.1",
                    markersize=3,
                    markevery=2,
                )
            k += 1


plt.xlabel("Communication Round", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title(f"dataset=Fashion-MNIST, alpha={args.alpha}, num_clients=100")
plt.grid("on")
plt.tight_layout()
plt.legend()
# plt.savefig(
#     os.path.join(
#         "figures", args.sampling_type + "_multi_q_alpha_" + str(args.alpha) + ".pdf"
#     )
# )
plt.show()
