#!/usr/bin/env python3
# simple_euler.py

import smfsb
import numpy as np
import matplotlib.pyplot as plt

lamb = 2
alpha = 1
mu = 0.1
sig = 0.2


def my_drift(x, t):
    return np.array([lamb - x[0] * x[1], alpha * (mu - x[1])])


def my_diff(x, t):
    return np.array([[np.sqrt(lamb + x[0] * x[1]), 0], [0, sig * np.sqrt(x[1])]])


step_proc = smfsb.step_sde(my_drift, my_diff, dt=0.001)
out = smfsb.sim_time_series(np.array([1, 0.1]), 0, 30, 0.01, step_proc)

fig, axis = plt.subplots()
for i in range(2):
    axis.plot(np.arange(0, 30, 0.01), out[:, i])
fig.savefig("step_sde.pdf")


# eof
