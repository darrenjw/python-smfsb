# metropolis_hastings.py

import smfsb
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

data = np.random.normal(5, 2, 250)


def llik(x):
    return np.sum(sp.stats.norm.logpdf(data, x[0], x[1]))


def prop(x):
    return np.random.normal(x, 0.1, 2)


postmat = smfsb.metropolis_hastings([1, 1], llik, prop, verb=False)

fig, axes = plt.subplots(3, 2)
axes[0, 0].scatter(postmat[:, 0], postmat[:, 1], s=0.5)
axes[0, 1].plot(postmat[:, 0], postmat[:, 1], linewidth=0.1)
axes[1, 0].plot(range(10000), postmat[:, 0], linewidth=0.1)
axes[1, 1].plot(range(10000), postmat[:, 1], linewidth=0.1)
axes[2, 0].hist(postmat[:, 0], bins=30)
axes[2, 1].hist(postmat[:, 1], bins=30)
fig.savefig("metropolis_hastings.pdf")


# eof
