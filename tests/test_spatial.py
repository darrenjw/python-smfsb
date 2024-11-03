# test_spatial.py
# tests relating to chapter 9

import smfsb
import numpy as np
import matplotlib.pyplot as plt
import smfsb.models

def test_step_gillespie1D():
    N=20
    x0 = np.zeros((2,N))
    lv = smfsb.models.lv()
    x0[:,int(N/2)] = lv.m
    stepLv1d = lv.step_gillespie1D(np.array([0.6, 0.6]))
    x1 = stepLv1d(x0, 0, 1)
    assert(x1.shape == (2,N))

def test_sim_time_series1D():
    N=8
    T=6
    x0 = np.zeros((2,N))
    lv = smfsb.models.lv()
    x0[:,int(N/2)] = lv.m
    stepLv1d = lv.step_gillespie1D(np.array([0.6, 0.6]))
    out = smfsb.sim_time_series1D(x0, 0, T, 1, stepLv1d)
    assert(out.shape == (2,N,T+1))

def test_step_cle1D():
    N=20
    x0 = np.zeros((2,N))
    lv = smfsb.models.lv()
    x0[:,int(N/2)] = lv.m
    stepLv1d = lv.step_cle1D(np.array([0.6, 0.6]))
    x1 = stepLv1d(x0, 0, 1)
    assert(x1.shape == (2,N))

def test_step_gillespie2D():
    M=16
    N=20
    x0 = np.zeros((2,M,N))
    lv = smfsb.models.lv()
    x0[:,int(M/2),int(N/2)] = lv.m
    stepLv2d = lv.step_gillespie2D(np.array([0.6, 0.6]))
    x1 = stepLv2d(x0, 0, 1)
    assert(x1.shape == (2,M,N))

def test_step_cle2D():
    M=16
    N=20
    x0 = np.zeros((2,M,N))
    lv = smfsb.models.lv()
    x0[:,int(M/2),int(N/2)] = lv.m
    stepLv2d = lv.step_cle2D(np.array([0.6, 0.6]))
    x1 = stepLv2d(x0, 0, 1)
    assert(x1.shape == (2,M,N))

def test_sim_time_series2D():
    M=16
    N=20
    x0 = np.zeros((2,M,N))
    lv = smfsb.models.lv()
    x0[:,int(M/2),int(N/2)] = lv.m
    stepLv2d = lv.step_cle2D(np.array([0.6, 0.6]))
    out = smfsb.sim_time_series2D(x0, 0, 5, 1, stepLv2d)
    assert(out.shape == (2,M,N,6))

    
# eof


