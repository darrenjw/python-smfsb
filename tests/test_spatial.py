# test_spatial.py
# tests relating to chapter 9

import smfsb
import numpy as np
import matplotlib.pyplot as plt
import smfsb.models

def test_stepGillespie1D():
    N=20
    x0 = np.zeros((2,N))
    lv = smfsb.models.lv()
    x0[:,int(N/2)] = lv.m
    stepLv1d = lv.stepGillespie1D(np.array([0.6, 0.6]))
    x1 = stepLv1d(x0, 0, 1)
    assert(x1.shape == (2,N))

def test_simTs1D():
    N=8
    T=6
    x0 = np.zeros((2,N))
    lv = smfsb.models.lv()
    x0[:,int(N/2)] = lv.m
    stepLv1d = lv.stepGillespie1D(np.array([0.6, 0.6]))
    out = smfsb.simTs1D(x0, 0, T, 1, stepLv1d)
    assert(out.shape == (2,N,T+1))

def test_stepCLE1D():
    N=20
    x0 = np.zeros((2,N))
    lv = smfsb.models.lv()
    x0[:,int(N/2)] = lv.m
    stepLv1d = lv.stepCLE1D(np.array([0.6, 0.6]))
    x1 = stepLv1d(x0, 0, 1)
    assert(x1.shape == (2,N))

def test_stepGillespie2D():
    M=16
    N=20
    x0 = np.zeros((2,M,N))
    lv = smfsb.models.lv()
    x0[:,int(M/2),int(N/2)] = lv.m
    stepLv2d = lv.stepGillespie2D(np.array([0.6, 0.6]))
    x1 = stepLv2d(x0, 0, 1)
    assert(x1.shape == (2,M,N))

def test_stepCLE2D():
    M=16
    N=20
    x0 = np.zeros((2,M,N))
    lv = smfsb.models.lv()
    x0[:,int(M/2),int(N/2)] = lv.m
    stepLv2d = lv.stepCLE2D(np.array([0.6, 0.6]))
    x1 = stepLv2d(x0, 0, 1)
    assert(x1.shape == (2,M,N))

def test_simTs2D():
    M=16
    N=20
    x0 = np.zeros((2,M,N))
    lv = smfsb.models.lv()
    x0[:,int(M/2),int(N/2)] = lv.m
    stepLv2d = lv.stepCLE2D(np.array([0.6, 0.6]))
    out = smfsb.simTs2D(x0, 0, 5, 1, stepLv2d)
    assert(out.shape == (2,M,N,6))

    
# eof


