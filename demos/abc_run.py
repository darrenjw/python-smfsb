# abc_run.py

import smfsb
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(5, 2, 250)


def rpr():
    return np.exp(np.random.uniform(-3, 3, 2))


def rmod(th):
    return np.random.normal(th[0], th[1], 250)


def sum_stats(dat):
    return np.array([np.mean(dat), np.std(dat)])


ssd = sum_stats(data)


def dist(ss):
    diff = ss - ssd
    return np.sqrt(np.sum(diff * diff))


def rdis(th):
    return dist(sum_stats(rmod(th)))


p, d = smfsb.abc_run(1000000, rpr, rdis)

q = np.quantile(d, 0.01)
prmat = np.vstack(p)
postmat = prmat[d < q, :]
its, var = postmat.shape

fig, axes = plt.subplots(3, 2)
axes[0, 0].scatter(postmat[:, 0], postmat[:, 1], s=0.5)
axes[0, 1].plot(postmat[:, 0], postmat[:, 1], linewidth=0.1)
axes[1, 0].plot(range(its), postmat[:, 0], linewidth=0.1)
axes[1, 1].plot(range(its), postmat[:, 1], linewidth=0.1)
axes[2, 0].hist(postmat[:, 0], bins=30)
axes[2, 1].hist(postmat[:, 1], bins=30)
fig.savefig("abc_run.pdf")


# eof
