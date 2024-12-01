# test_early.py
# test of code from the early part of the book

import smfsb
import numpy as np


def test_imdeath():
    times, states = smfsb.imdeath(150)
    assert len(times) == 150
    assert len(states) == 151


def test_rcfmc():
    q_mat = np.array([[-0.5, 0.5], [1, -1]])
    pi0 = np.array([0.5, 0.5])
    times, states = smfsb.rcfmc(30, q_mat, pi0)
    assert len(times) == 30
    assert len(states) == 31


def test_rfmc():
    p_mat = np.array([[0.9, 0.1], [0.2, 0.8]])
    pi0 = np.array([0.5, 0.5])
    out = smfsb.rfmc(200, p_mat, pi0)
    assert len(out) == 200
    assert out[100] >= 0


def lv(th=[1, 0.1, 0.1]):
    def rhs(x, t):
        return np.array(
            [th[0] * x[0] - th[1] * x[0] * x[1], th[1] * x[0] * x[1] - th[2] * x[1]]
        )

    return rhs


def test_simple_euler():
    out = smfsb.simple_euler(lv(), np.array([4, 10]), 100)
    assert out[99, 1] > 0.0


def test_rdiff():
    out = smfsb.rdiff(lambda x: 1 - 0.1 * x, lambda x: np.sqrt(1 + 0.1 * x))
    assert len(out) > 500
    assert out[500] >= 0.0


lamb = 2
alpha = 1
mu = 0.1
sig = 0.2


def my_drift(x, t):
    return np.array([lamb - x[0] * x[1], alpha * (mu - x[1])])


def my_diff(x, t):
    return np.array([[np.sqrt(lamb + x[0] * x[1]), 0], [0, sig * np.sqrt(x[1])]])


def test_step_sde():
    step_proc = smfsb.step_sde(my_drift, my_diff, dt=0.001)
    out = smfsb.sim_time_series(np.array([1, 0.1]), 0, 30, 0.01, step_proc)
    assert out.shape == (3000, 2)
    assert out[1000, 0] >= 0.0
    assert out[1000, 1] >= 0.0


# eof
