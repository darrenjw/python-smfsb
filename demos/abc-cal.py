# abc-cal.py
# ABC with calibrated summary stats

import smfsb
import numpy as np
import math
import matplotlib.pyplot as plt

print("ABC with calibrated summary stats")

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


def ss1d(vec):
    n = len(vec)
    mean = np.nanmean(vec)
    v0 = vec - mean
    var = np.nanvar(v0)
    acs = [
        np.corrcoef(v0[range(n - 1)], v0[range(1, n)])[0, 1],
        np.corrcoef(v0[range(n - 2)], v0[range(2, n)])[0, 1],
        np.corrcoef(v0[range(n - 3)], v0[range(3, n)])[0, 1],
    ]
    # print(mean)
    # print(var)
    # print(acs)
    return np.array([np.log(mean + 1), np.log(var + 1), acs[0], acs[1], acs[2]])


def ssi(ts):
    return np.concatenate(
        (ss1d(ts[:, 0]), ss1d(ts[:, 1]), [np.corrcoef(ts[:, 0], ts[:, 1])[0, 1]])
    )


print("Pilot run")

p, d = smfsb.abc_run(100000, rpr, lambda th: ssi(rmod(th)), verb=True)
prmat = np.vstack(p)
dmat = np.vstack(d)
print(prmat.shape)
print(dmat.shape)
dmat[dmat == math.inf] = math.nan
sds = np.nanstd(dmat, 0)
print(sds)


print("Main run with calibrated summary stats")


def sum_stats(dat):
    return ssi(dat) / sds


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
fig.savefig("abc-cal.pdf")

print("All done.")


# eof
