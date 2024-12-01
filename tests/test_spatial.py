# test_spatial.py
# tests relating to chapter 9

import smfsb
import numpy as np
import smfsb.models


def test_step_gillespie_1d():
    n = 20
    x0 = np.zeros((2, n))
    lv = smfsb.models.lv()
    x0[:, int(n / 2)] = lv.m
    step_lv_1d = lv.step_gillespie_1d(np.array([0.6, 0.6]))
    x1 = step_lv_1d(x0, 0, 1)
    assert x1.shape == (2, n)


def test_sim_time_series_1d():
    n = 8
    t = 6
    x0 = np.zeros((2, n))
    lv = smfsb.models.lv()
    x0[:, int(n / 2)] = lv.m
    step_lv_1d = lv.step_gillespie_1d(np.array([0.6, 0.6]))
    out = smfsb.sim_time_series_1d(x0, 0, t, 1, step_lv_1d)
    assert out.shape == (2, n, t + 1)


def test_step_cle_1d():
    n = 20
    x0 = np.zeros((2, n))
    lv = smfsb.models.lv()
    x0[:, int(n / 2)] = lv.m
    step_lv_1d = lv.step_cle_1d(np.array([0.6, 0.6]))
    x1 = step_lv_1d(x0, 0, 1)
    assert x1.shape == (2, n)


def test_step_gillespie_2d():
    m = 16
    n = 20
    x0 = np.zeros((2, m, n))
    lv = smfsb.models.lv()
    x0[:, int(m / 2), int(n / 2)] = lv.m
    step_lv_2d = lv.step_gillespie_2d(np.array([0.6, 0.6]))
    x1 = step_lv_2d(x0, 0, 1)
    assert x1.shape == (2, m, n)


def test_step_cle_2d():
    m = 16
    n = 20
    x0 = np.zeros((2, m, n))
    lv = smfsb.models.lv()
    x0[:, int(m / 2), int(n / 2)] = lv.m
    step_lv_2d = lv.step_cle_2d(np.array([0.6, 0.6]))
    x1 = step_lv_2d(x0, 0, 1)
    assert x1.shape == (2, m, n)


def test_sim_time_series_2d():
    m = 16
    n = 20
    x0 = np.zeros((2, m, n))
    lv = smfsb.models.lv()
    x0[:, int(m / 2), int(n / 2)] = lv.m
    step_lv_2d = lv.step_cle_2d(np.array([0.6, 0.6]))
    out = smfsb.sim_time_series_2d(x0, 0, 5, 1, step_lv_2d)
    assert out.shape == (2, m, n, 6)


# eof
