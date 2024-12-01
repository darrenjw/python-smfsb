# test_inference.py
# tests relating to chapters 10 and 11

import smfsb
import numpy as np
import scipy as sp


def test_normal_gibbs():
    postmat = smfsb.normal_gibbs(100, 15, 3, 11, 10, 1 / 100, 25, 20)
    assert postmat.shape == (100, 2)


def test_metrop():
    vec = smfsb.metrop(1000, 1)
    assert len(vec) == 1000


def test_metropolis_hastings():
    data = np.random.normal(5, 2, 250)

    def llik(x):
        return np.sum(sp.stats.norm.logpdf(data, x[0], x[1]))

    def prop(x):
        return np.random.normal(x, 0.1, 2)

    out = smfsb.metropolis_hastings([1, 1], llik, prop, iters=1000, thin=2, verb=False)
    assert out.shape == (1000, 2)


def test_abc_run():
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

    p, d = smfsb.abc_run(100, rpr, rdis)
    assert len(p) == 100
    assert len(d) == 100


def test_pfmllik():
    def obsll(x, t, y, th):
        return np.sum(sp.stats.norm.logpdf((y - x) / 10))

    def sim_x(t0, th):
        return np.array([np.random.poisson(50), np.random.poisson(100)])

    def step(x, t, dt, th):
        sf = smfsb.models.lv(th).step_cle()
        return sf(x, t, dt)

    mll = smfsb.pf_marginal_ll(50, sim_x, 0, step, obsll, smfsb.data.lv_noise_10)
    assert mll(np.array([1, 0.005, 0.6])) > mll(np.array([2, 0.005, 0.6]))


def test_abcsmcstep():
    data = np.random.normal(5, 2, 250)

    def rpr():
        return np.exp(np.random.uniform(-3, 3, 2))

    def rmod(th):
        return np.random.normal(np.exp(th[0]), np.exp(th[1]), 250)

    def sum_stats(dat):
        return np.array([np.mean(dat), np.std(dat)])

    ssd = sum_stats(data)

    def dist(ss):
        diff = ss - ssd
        return np.sqrt(np.sum(diff * diff))

    def rdis(th):
        return dist(sum_stats(rmod(th)))

    num_samples = 100
    samples = np.zeros((num_samples, 1))
    samples = np.apply_along_axis(lambda x: rpr(), 1, samples)
    th, lw = smfsb.abc_smc_step(
        lambda x: np.log(np.sum(((x < 3) & (x > -3)) / 6)),
        samples,
        np.zeros(num_samples) + np.log(1 / num_samples),
        rdis,
        lambda x: np.random.normal(x, 0.1),
        lambda x, y: np.sum(sp.stats.norm(x, 0.1).logpdf(y)),
        10,
    )
    assert th.shape == (num_samples, 2)
    assert len(lw) == num_samples


def test_abcsmc():
    data = np.random.normal(5, 2, 250)

    def rpr():
        return np.exp(np.random.uniform(-3, 3, 2))

    def rmod(th):
        return np.random.normal(np.exp(th[0]), np.exp(th[1]), 250)

    def sum_stats(dat):
        return np.array([np.mean(dat), np.std(dat)])

    ssd = sum_stats(data)

    def dist(ss):
        diff = ss - ssd
        return np.sqrt(np.sum(diff * diff))

    def rdis(th):
        return dist(sum_stats(rmod(th)))

    num_samples = 100
    post = smfsb.abc_smc(
        num_samples,
        rpr,
        lambda x: np.log(np.sum(((x < 3) & (x > -3)) / 6)),
        rdis,
        lambda x: np.random.normal(x, 0.1),
        lambda x, y: np.sum(sp.stats.norm.logpdf(y, x, 0.1)),
    )
    assert post.shape == (num_samples, 2)


# eof
