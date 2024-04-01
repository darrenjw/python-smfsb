# pmmh.py
# PMMH demo
# Particle marginal Metropolis-Hastings

import smfsb
import scipy as sp
import numpy as np

print("PMMH")

def obsll(x, t, y, th):
    return np.sum(sp.stats.norm.logpdf(y-x, scale=10))
def simX(t0, th):
    return np.array([np.random.poisson(50), np.random.poisson(100)])
def step(x, t, dt, th):
    sf = smfsb.models.lv(th).stepGillespie()
    #sf = smfsb.models.lv(th).stepCLE(0.1)
    return sf(x, t, dt)
mll = smfsb.pfMLLik(100, simX, 0, step, obsll, smfsb.data.LVnoise10)

print("Test evals")

print(mll(np.array([1, 0.005, 0.6])))
print(mll(np.array([1, 0.005, 0.5])))

print("Now the main MCMC loop")

def prop(th, tune=0.01):
    return th*np.exp(np.random.normal(0, tune, (3)))

thmat = smfsb.metropolisHastings([1, 0.005, 0.6], mll, prop,
                                 iters=10000, thin=1, verb=True)

print("MCMC done. Now processing the results...")

def mcmcSummary(mat, fileName="mcmc.pdf"):
    import matplotlib.pyplot as plt
    n, p = mat.shape
    fig, axes = plt.subplots(p, 2)
    for i in range(p):
        axes[i, 0].plot(range(n), mat[:,i], linewidth=0.1)
        axes[i, 1].hist(mat[:,i], bins=30)
    fig.savefig(fileName)

mcmcSummary(thmat, "pmmh.pdf")

print("All finished.")

# eof

