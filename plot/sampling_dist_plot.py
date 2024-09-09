import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from shared.client_sampling import client_sampling

sampling_type = "UGamma"
allclients = []
for _ in range(1000):
  allclients.extend(client_sampling(sampling_type=sampling_type.lower(), clients=list(range(100)), round=None))

print(len(list(set(allclients))))

y = np.linspace(0,100, num=100)
fig, ax = plt.subplots(figsize=(6, 5))
sns.histplot(np.array(allclients), ax=ax, bins=np.arange(0, 101, 1))
plt.xlabel("Clients", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title(f"Client Participation under {sampling_type} Distribution")
plt.tight_layout()
plt.savefig(
    os.path.join(
        "figures", "sampling_dist", f"{sampling_type}.pdf"
    )
)
plt.show()


# sampled_clients = np.random.weibull(a=20, size=50000)
# x = np.linspace(0,100, num=100)
# fig, ax = plt.subplots()
# sns.histplot(np.array(sampled_clients), ax=ax)
# plt.xlabel("Clients", fontsize=12)
# plt.ylabel("Count", fontsize=12)
# plt.title(f"Client Participation under Weibull Distribution")
# plt.show()