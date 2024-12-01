# pf_marginal_ll.py

import smfsb
import scipy as sp
import numpy as np


def obsll(x, t, y, th):
    return np.sum(sp.stats.norm.logpdf(y - x, scale=10))


def sim_x(t0, th):
    return np.array([np.random.poisson(50), np.random.poisson(100)])


def step(x, t, dt, th):
    sf = smfsb.models.lv(th).step_cle(0.1)
    # sf = smfsb.models.lv(th).step_gillespie()
    return sf(x, t, dt)


mll = smfsb.pf_marginal_ll(100, sim_x, 0, step, obsll, smfsb.data.lv_noise_10)

print(mll(np.array([1, 0.005, 0.6])))
print(mll(np.array([1, 0.005, 0.6])))
print(mll(np.array([1, 0.005, 0.6])))


print(mll(np.array([1, 0.005, 0.5])))
print(mll(np.array([1, 0.005, 0.5])))
