import os
import pandas as pd
from config import get_parms


args = get_parms("Stats").parse_args()
dataset = "fashion"
if "uniform_" in args.sampling_type:
    sampling_type = args.sampling_type.replace("uniform_", "")
else:
    sampling_type = args.sampling_type


filenames = os.listdir(
    f"results/{dataset}_{args.algo}/{sampling_type}_alpha_{args.alpha}/"
)
# filenames = os.listdir(f"results/{dataset}_fedprox/alpha_{alpha}/")
for filename in filenames:
    print(filename)
    df = pd.read_csv(
        f"results/{dataset}_{args.algo}/{sampling_type}_alpha_{args.alpha}/{filename}"
    )
    # df = pd.read_csv(f"results/{dataset}_fedprox/alpha_{alpha}/{filename}")
    array = df.iloc[len(df.index) - 5 : len(df.index), 2]
    print(f"{array.mean()*100 =:.2f}")
    print(f"{array.std()*100 =:.1f}")
    print()
