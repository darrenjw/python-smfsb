# abc.py

import smfsb
import numpy as np
import matplotlib.pyplot as plt

print("ABC")

data = smfsb.data.lv_perfect[:, range(1, 3)]


def rpr():
    return np.exp(
        np.array(
            [
                np.random.uniform(-3, 3),
                np.random.uniform(-8, -2),
                np.random.uniform(-4, 2),
            ]
        )
    )


def rmod(th):
    return smfsb.sim_time_series([50, 100], 0, 30, 2, smfsb.models.lv(th).step_cle(0.1))


def sum_stats(dat):
    return dat


ssd = sum_stats(data)


def dist(ss):
    diff = ss - ssd
    return np.sqrt(np.sum(diff * diff))


def rdis(th):
    return dist(sum_stats(rmod(th)))


p, d = smfsb.abc_run(1000000, rpr, rdis, verb=True)

q = np.nanquantile(d, 0.01)
prmat = np.vstack(p)
postmat = prmat[d < q, :]
its, var = postmat.shape
print(its, var)

postmat = np.log(postmat)  # look at posterior on log scale

fig, axes = plt.subplots(2, 3)
axes[0, 0].scatter(postmat[:, 0], postmat[:, 1], s=0.5)
axes[0, 1].scatter(postmat[:, 0], postmat[:, 2], s=0.5)
axes[0, 2].scatter(postmat[:, 1], postmat[:, 2], s=0.5)
axes[1, 0].hist(postmat[:, 0], bins=30)
axes[1, 1].hist(postmat[:, 1], bins=30)
axes[1, 2].hist(postmat[:, 2], bins=30)
fig.savefig("abc.pdf")

print("All done.")


# eof
