# abc_smc.py

import smfsb
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt


print("ABC-SMC")

data = smfsb.data.lv_perfect[:, range(1, 3)]


def rpr():
    return np.array(
        [np.random.uniform(-2, 2), np.random.uniform(-7, -3), np.random.uniform(-3, 1)]
    )


def dpr(th):
    return np.sum(
        np.log(
            np.array(
                [
                    ((th[0] > -2) & (th[0] < 2)) / 4,
                    ((th[1] > -7) & (th[1] < -3)) / 4,
                    ((th[2] > -3) & (th[2] < 1)) / 4,
                ]
            )
        )
    )


def rmod(th):
    return smfsb.sim_time_series(
        [50, 100], 0, 30, 2, smfsb.models.lv(np.exp(th)).step_cle(0.1)
    )


print("Pilot run...")


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


p, d = smfsb.abc_run(20000, rpr, lambda th: ssi(rmod(th)), verb=False)
prmat = np.vstack(p)
dmat = np.vstack(d)
print(prmat.shape)
print(dmat.shape)
dmat[dmat == math.inf] = math.nan
sds = np.nanstd(dmat, 0)
print(sds)


def sum_stats(dat):
    return ssi(dat) / sds


ssd = sum_stats(data)

print("Main ABC-SMC run")


def dist(ss):
    diff = ss - ssd
    return np.sqrt(np.sum(diff * diff))


def rdis(th):
    return dist(sum_stats(rmod(th)))


def rper(th):
    return th + np.random.normal(0, 0.5, 3)


def dper(ne, ol):
    return np.sum(sp.stats.norm.logpdf(ne, ol, 0.5))


postmat = smfsb.abc_smc(
    10000, rpr, dpr, rdis, rper, dper, factor=5, steps=8, verb=True, debug=True
)

its, var = postmat.shape
print(its, var)

fig, axes = plt.subplots(2, 3)
axes[0, 0].scatter(postmat[:, 0], postmat[:, 1], s=0.5)
axes[0, 1].scatter(postmat[:, 0], postmat[:, 2], s=0.5)
axes[0, 2].scatter(postmat[:, 1], postmat[:, 2], s=0.5)
axes[1, 0].hist(postmat[:, 0], bins=30)
axes[1, 1].hist(postmat[:, 1], bins=30)
axes[1, 2].hist(postmat[:, 2], bins=30)
fig.savefig("abc_smc.pdf")

print("All done.")


# eof
