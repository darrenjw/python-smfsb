# test_spn.py
# tests of basic Spn construction and functionality

import smfsb
import numpy as np


def test_create():
    sir = smfsb.Spn(
        ["S", "I", "R"],
        ["S->I", "I->R"],
        [[1, 1, 0], [0, 1, 0]],
        [[0, 2, 0], [0, 0, 1]],
        lambda x, t: np.array([0.3 * x[0] * x[1] / 200, 0.1 * x[1]]),
        [197, 3, 0],
    )
    step_sir = sir.step_poisson()
    sample = smfsb.sim_sample(20, sir.m, 0, 20, step_sir)
    assert sample[19, 1] >= 0


def test_bd():
    bd = smfsb.models.bd()
    step = bd.step_gillespie()
    out = smfsb.sim_time_series(bd.m, 0, 100, 0.1, step)
    assert out.shape == (1000, 1)
    assert out[100, 0] >= 0


def test_dimer():
    dimer = smfsb.models.dimer()
    step = dimer.step_gillespie()
    out = smfsb.sim_time_series(dimer.m, 0, 100, 0.1, step)
    assert out.shape == (1000, 2)
    assert out[600, 1] >= 0


def test_id():
    id = smfsb.models.id()
    step = id.step_gillespie()
    out = smfsb.sim_time_series(id.m, 0, 100, 0.1, step)
    assert out.shape == (1000, 1)
    assert out[100, 0] >= 0


def test_lv2():
    lv = smfsb.models.lv([0.2, 0.001, 0.1])
    step_lv = lv.step_gillespie()
    out = smfsb.sim_time_series(lv.m, 0, 100, 0.1, step_lv)
    assert out.shape == (1000, 2)
    assert out[600, 1] >= 0


def test_cle():
    lv = smfsb.models.lv()
    step_lv = lv.step_cle()
    out = smfsb.sim_time_series(lv.m, 0, 100, 0.1, step_lv)
    assert out.shape == (1000, 2)
    assert out[600, 1] >= 0.0


def test_discretise():
    dt = 0.01
    lv = smfsb.models.lv()
    times, states = lv.gillespie(2000)
    out = smfsb.discretise(times, states, dt)
    assert out.shape[0] > 10
    assert out.shape[1] == 2
    assert out[10, 1] >= 0


def test_euler():
    lv = smfsb.models.lv()
    step_lv = lv.step_euler(0.001)
    out = smfsb.sim_time_series(lv.m, 0, 100, 0.1, step_lv)
    assert out.shape == (1000, 2)
    assert out[600, 1] > 0.0


def test_frm():
    lv = smfsb.models.lv()
    step_lv = lv.step_first()
    out = smfsb.sim_time_series(lv.m, 0, 10, 0.01, step_lv)
    assert out.shape == (1000, 2)
    assert out[600, 1] >= 0


def test_gillespied():
    lv = smfsb.models.lv()
    states = lv.gillespied(30, 0.1)
    assert states.shape == (300, 2)
    assert states[100, 1] >= 0


def test_gillespie():
    lv = smfsb.models.lv()
    times, states = lv.gillespie(2000)
    assert states.shape == (2001, 2)
    assert len(times) == 2000


def test_pts():
    lv = smfsb.models.lv()
    step_lv = lv.step_poisson()
    out = smfsb.sim_time_series(lv.m, 0, 100, 0.1, step_lv)
    assert out.shape == (1000, 2)
    assert out[600, 1] >= 0


def test_lv():
    lv = smfsb.models.lv()
    step_lv = lv.step_gillespie()
    out = smfsb.sim_time_series(lv.m, 0, 10, 0.01, step_lv)
    assert out.shape == (1000, 2)
    assert out[600, 1] >= 0


def test_mm():
    mm = smfsb.models.mm()
    step = mm.step_gillespie()
    out = smfsb.sim_time_series(mm.m, 0, 100, 0.1, step)
    assert out.shape == (1000, 4)
    assert out[600, 1] >= 0


def test_sir():
    sir = smfsb.models.sir()
    step = sir.step_gillespie()
    out = smfsb.sim_time_series(sir.m, 0, 100, 0.1, step)
    assert out.shape == (1000, 3)
    assert out[600, 1] >= 0


# eof
