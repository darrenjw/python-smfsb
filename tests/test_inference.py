# test_inference.py
# tests relating to chapters 10 and 11

import smfsb
import numpy as np
import scipy as sp

def test_normgibbs():
    postmat = smfsb.normgibbs(100, 15, 3, 11, 10, 1/100, 25, 20)
    assert(postmat.shape == (100, 2))

def test_metrop():
    vec = smfsb.metrop(1000, 1)
    assert(len(vec) == 1000)

def test_metropolisHastings():
    data = np.random.normal(5, 2, 250)
    llik = lambda x: np.sum(sp.stats.norm.logpdf(data, x[0], x[1]))
    prop = lambda x: np.random.normal(x, 0.1, 2)
    out = smfsb.metropolisHastings([1,1], llik, prop, iters=1000, thin=2, verb=False)
    assert(out.shape == (1000, 2))
    
# eof


