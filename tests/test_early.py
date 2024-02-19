# test_early.py
# test of code from the early part of the book

import smfsb
import numpy as np


def test_imdeath():
    times, states = smfsb.imdeath(150)
    assert(len(times) == 150)
    assert(len(states) == 151)

def test_rcfmc():
    Q = np.array([[-0.5,0.5],[1,-1]])
    pi0 = np.array([0.5,0.5])
    times, states = smfsb.rcfmc(30, Q, pi0)
    assert(len(times) == 30)
    assert(len(states) == 31)

def test_rdiff():
    out = smfsb.rdiff(lambda x: 1 - 0.1*x, lambda x: np.sqrt(1 + 0.1*x))
    assert(out[500] > 0.0)

def test_rfmc():
    P = np.array([[0.9,0.1],[0.2,0.8]])
    pi0 = np.array([0.5,0.5])
    out = smfsb.rfmc(200, P, pi0)
    assert(len(out) == 200)
    assert(out[100] >= 0)




    

def lv(th=[1, 0.1, 0.1]):
    def rhs(x, t):
        return np.array([th[0]*x[0] - th[1]*x[0]*x[1],
                         th[1]*x[0]*x[1] - th[2]*x[1]])
    return rhs

def test_simpleEuler():
    out = smfsb.simpleEuler(lv(), np.array([4, 10]), 100)
    assert(out[99, 1] > 0.0)

def test_rdiff():
    out = smfsb.rdiff(lambda x: 1 - 0.1*x, lambda x: np.sqrt(1 + 0.1*x))
    assert(len(out) > 500)
    assert(out[500] >= 0.0)

    
lamb = 2; alpha = 1; mu = 0.1; sig = 0.2

def myDrift(x, t):
    return np.array([lamb - x[0]*x[1],
                     alpha*(mu - x[1])])

def myDiff(x, t):
    return np.array([[np.sqrt(lamb + x[0]*x[1]), 0],
                     [0 ,sig*np.sqrt(x[1])]])

def test_stepSDE():
    stepProc = smfsb.stepSDE(myDrift, myDiff, dt=0.001)
    out = smfsb.simTs(np.array([1, 0.1]), 0, 30, 0.01, stepProc)
    assert(out.shape == (3000, 2))
    assert(out[1000,0] >= 0.0)
    assert(out[1000,1] >= 0.0)





    


# eof
