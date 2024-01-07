#!/usr/bin/env python3
# smfsb.py

import numpy as np

# Class for SPN models

class Spn:
    
    def __init__(self, n, t, pre, post, h, m):
        self.n = n # species names
        self.t = t # reaction names
        self.pre = np.matrix(pre)
        self.post = np.matrix(post)
        self.h = h # hazard function
        self.m = np.array(m) # initial marking
        
    def __str__(self):
        return "n: {}\n t: {}\npre: {}\npost: {}\nh: {}\nm: {}".format(str(self.n),
                str(self.t), str(self.pre), str(self.post), str(self.h), str(self.m))

    def stepGillespie(self):
        S = (self.post - self.pre).T
        u, v = S.shape
        def step(x0, t0, deltat):
            t = t0
            x = x0
            termt = t0 + deltat
            while(True):
                h = self.h(x, t)
                h0 = h.sum()
                if (h0 > 1e07):
                    print("WARNING: hazard too large - terminating!")
                    return(x)
                if (h0 < 1e-10):
                    t = 1e99
                else:
                    t = t + np.random.exponential(1.0/h0)
                if (t > termt):
                    return(x)
                j = np.random.choice(v, p=h/h0)
                x = np.add(x, S[:,j].A1)
        return step

    def stepPTS(self, dt = 0.01):
        S = (self.post - self.pre).T
        u, v = S.shape
        def step(x0, t0, deltat):
            x = x0
            t = t0
            termt = t0 + deltat
            while(True):
                h = self.h(x, t)
                r = np.random.poisson(h * dt)
                x = np.add(x, S.dot(r).A1)
                t = t + dt
                if (t > termt):
                    return x
        return step
    
    def stepEuler(self, dt = 0.01):
        S = (self.post - self.pre).T
        def step(x0, t0, deltat):
            x = x0
            t = t0
            termt = t0 + deltat
            while(True):
                h = self.h(x, t)
                x = np.add(x, S.dot(h*dt).A1)
                t = t + dt
                if (t > termt):
                    return x
        return step


    
# Some simulation functions

def simTs(x0, t0, tt, dt, stepFun):
    n = int((tt-t0) // dt) + 1
    u = len(x0)
    mat = np.zeros((n, u))
    x = x0
    t = t0
    mat[1,:] = x
    for i in range(n):
        t = t + dt
        x = stepFun(x, t, dt)
        mat[i,:] = x
    return mat

def simSample(n, x0, t0, deltat, stepFun):
    u = len(x0)
    mat = np.zeros((n, u))
    for i in range(n):
        mat[i,:] = stepFun(x0, t0, deltat)
    return mat



# Misc utility functions

import inspect

def showSource(fun):
    print(inspect.getsource(fun))




# eof

