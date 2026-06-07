# pf_marginal_ll.py

import smfsb
import scipy as sp
import numpy as np


def obsll(x, t, y, th):
    return np.sum(sp.stats.norm.logpdf(y - x, scale=10))


def sim_x(rng, t0, th):
    return np.array([rng.poisson(50), rng.poisson(100)])


def step(rng, x, t, dt, th):
    sf = smfsb.models.lv(th).step_cle(0.1)
    return sf(rng, x, t, dt)


mll = smfsb.pf_marginal_ll(100, sim_x, 0, step, obsll, smfsb.data.lv_noise_10)

rng = np.random.default_rng()

print(mll(rng, np.array([1, 0.005, 0.6])))
print(mll(rng, np.array([1, 0.005, 0.6])))
print(mll(rng, np.array([1, 0.005, 0.6])))

print(mll(rng, np.array([1, 0.005, 0.5])))
print(mll(rng, np.array([1, 0.005, 0.5])))
