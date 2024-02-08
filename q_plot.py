import os
import dataclasses
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import argparse
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

parser = argparse.ArgumentParser(description="Q Plot")
parser.add_argument(
    "--sampling-type",
    type=str,
    default="uniform_gamma",
)
parser.add_argument("--alpha", type=float, default=0.05)
args = parser.parse_args()


@dataclasses.dataclass
class Params:
    sampling_type: str
    alpha: float
    gamma: int


def parse_filename(filename: str) -> Params:
    tokens = filename.replace(".csv", "").split(",")
    params = Params(
        sampling_type=tokens[0].replace("list_q_", ""),
        alpha=float(tokens[1].replace("alpha=", "")),
        gamma=int(tokens[2].replace("lambda=", "").replace(".csv", "")),
    )
    return params


fig, ax = plt.subplots(figsize=(6, 5))
plt.rc("font", size=12)
markers = "os*xdXDHhPp12348<>"
k = 0
filenames = os.listdir(f"results/q/")

df_all = pd.DataFrame()
for filename in [
    "list_q_uniform_weibull,alpha=0.05,lambda=7.csv",
    "list_q_uniform_weibull,alpha=0.05,lambda=1.csv",
]:
    params = parse_filename(filename)
    if args.sampling_type in params.sampling_type:
        df = pd.read_csv(f"results/q/{filename}")
        df["Lambda"] = f"{params.gamma}"
        df_all = pd.concat([df_all, df])

sns.histplot(df_all, x="Value", hue="Lambda", ax=ax, kde=True, bins=50)
plt.title(f"Fashion-MNIST, Alpha={args.alpha}, {args.sampling_type}")
plt.xlabel("q", fontsize=12)
plt.savefig(
    os.path.join(
        "figures",
        "q",
        f"fashion_{args.sampling_type}_adaptive q_alpha={str(args.alpha)}_bar.pdf",
    )
)
plt.show()
