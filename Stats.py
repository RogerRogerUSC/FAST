import numpy as np
import os
import pandas as pd

dataset = "fashion"
alpha = "0.1"
sample_type = "weibull"


# filenames = os.listdir("results/"+dataset+"/alpha_"+str(alpha)+"/")
# for filename in filenames: 
#     print(filename)
#     df = pd.read_csv("results/"+dataset+"/alpha_"+str(alpha)+"/" + filename)
#     array = df.iloc[len(df.index)-5:len(df.index),2]
#     print(f"{array.mean()*100 =:.2f}")
#     print(f"{array.std()*100 =:.1f}")
#     print()

# for filename in filenames:
#     print(filename)
#     df = pd.read_csv("results/"+dataset+"/alpha_"+str(alpha)+"/" + filename)
#     array = df["Value"]
#     print(array.max())
#     print()

filenames = os.listdir(f"results/{dataset}/{sample_type}_alpha_{alpha}/")
for filename in filenames: 
    print(filename)
    df = pd.read_csv(f"results/{dataset}/{sample_type}_alpha_{alpha}/{filename}")
    array = df.iloc[len(df.index)-5:len(df.index),2]
    print(f"{array.mean()*100 =:.2f}")
    print(f"{array.std()*100 =:.1f}")
    print()

