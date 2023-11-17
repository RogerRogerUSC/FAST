import numpy as np
import os
import pandas as pd

dataset = "mnist"
alpha = 0.3


filenames = os.listdir("results/"+dataset+"/alpha_"+str(alpha)+"/")
for filename in filenames: 
    print(filename)
    df = pd.read_csv("results/"+dataset+"/alpha_"+str(alpha)+"/" + filename)
    array = df.iloc[len(df.index)-5:len(df.index),2]
    print(f"{array.mean()*100 =:.2f}")
    print(f"{array.std()*100 =:.1f}")
    print()

# for filename in filenames:
#     print(filename)
#     df = pd.read_csv("results/"+dataset+"/alpha_"+str(alpha)+"/" + filename)
#     array = df["Value"]
#     print(array.max())
#     print()


