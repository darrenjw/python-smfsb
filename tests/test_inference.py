# test_inference.py
# tests relating to chapters 10 and 11

import smfsb
import numpy as np

def test_normgibbs():
    postmat = smfsb.normgibbs(100, 15, 3, 11, 10, 1/100, 25, 20)
    assert(postmat.shape == (100, 2))

def test_metrop():
    vec = smfsb.metrop(1000, 1)
    assert(len(vec) == 1000)

# eof


