#!/usr/bin/env python3

# Simulate a Lotka-Volterra deterministically using a simple Euler method

import smfsb
import smfsb.models
import matplotlib.pyplot as plt

lv = smfsb.models.lv()
print(lv)
step_lv = lv.step_euler(0.001)
out = smfsb.sim_time_series(None, lv.m, 0, 100, 0.1, step_lv)
# Don't acutally need to pass in a genuine random number generator, since
#  the Euler method doesn't use it.

fig, axis = plt.subplots()
for i in range(2):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(lv.n)
fig.savefig("s-m-lv-euler.pdf")

# eof
