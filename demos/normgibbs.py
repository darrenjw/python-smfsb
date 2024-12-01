# normal_gibbs.py

import smfsb
import numpy as np
import matplotlib.pyplot as plt

postmat = smfsb.normal_gibbs(11000, 15, 3, 11, 10, 1 / 100, 25, 20)
postmat = postmat[range(1000, 11000), :]


fig, axes = plt.subplots(3, 3)
axes[0, 0].scatter(postmat[:, 0], postmat[:, 1], s=0.5)
axes[0, 1].plot(postmat[:, 0], postmat[:, 1], linewidth=0.1)
axes[1, 0].plot(range(10000), postmat[:, 0], linewidth=0.1)
axes[1, 1].plot(range(10000), postmat[:, 1], linewidth=0.1)
axes[1, 2].plot(range(10000), 1 / np.sqrt(postmat[:, 1]), linewidth=0.1)
axes[2, 0].hist(postmat[:, 0], bins=30)
axes[2, 1].hist(postmat[:, 1], bins=30)
axes[2, 2].hist(1 / np.sqrt(postmat[:, 1]), bins=30)
fig.savefig("normal_gibbs.pdf")


# eof
