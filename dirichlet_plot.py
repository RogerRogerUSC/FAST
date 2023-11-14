"""
    Author: Bicheng Ying
    Description: This file is used to show the Dirichlet Distribution 
                 with different parameters. 
"""

import numpy as np
import matplotlib.pyplot as plt

data_dirichlet_alpha = 0.01
num_classes = 10
num_clients = 50
data_dist = np.random.dirichlet(
    data_dirichlet_alpha * np.ones(num_classes), num_clients
)

left = 0
for i in range(num_classes):
    plt.barh(range(num_clients), data_dist.transpose()[i], left=left, color=f"C{i}")
    left += data_dist.transpose()[i]
plt.title("alpha=" + str(data_dirichlet_alpha))
plt.ylabel("Clients", fontsize=12)
plt.show()

# s = np.random.dirichlet((10, 5, 3), 20).transpose()

# plt.barh(range(20), s[0])
# plt.barh(range(20), s[1], left=s[0], color="g")
# plt.barh(range(20), s[2], left=s[0] + s[1], color="r")
# plt.title("Lengths of Strings")
