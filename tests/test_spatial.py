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


    
# eof


