# pmmh.py
# PMMH demo
# Particle marginal Metropolis-Hastings

import smfsb
import mcmc
import scipy as sp
import numpy as np

print("PMMH")


def obsll(x, t, y, th):
    return np.sum(sp.stats.norm.logpdf(y - x, scale=10))


def sim_x(t0, th):
    return np.array([np.random.poisson(50), np.random.poisson(100)])


def step(x, t, dt, th):
    # sf = smfsb.models.lv(th).step_gillespie()
    sf = smfsb.models.lv(th).step_cle(0.1)
    return sf(x, t, dt)


mll = smfsb.pf_marginal_ll(100, sim_x, 0, step, obsll, smfsb.data.lv_noise_10)

print("Test evals")

print(mll(np.array([1, 0.005, 0.6])))
print(mll(np.array([1, 0.005, 0.5])))

print("Now the main MCMC loop")


def prop(th, tune=0.01):
    return th * np.exp(np.random.normal(0, tune, (3)))


thmat = smfsb.metropolis_hastings(
    [1, 0.005, 0.6], mll, prop, iters=5000, thin=1, verb=True
)

print("MCMC done. Now processing the results...")

mcmc.mcmc_summary(thmat, "pmmh.pdf")

print("All finished.")

# eof
