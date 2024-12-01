#!/usr/bin/env python3

import smfsb
import smfsb.models
import numpy as np
import matplotlib.pyplot as plt


dt = 0.01

lv = smfsb.models.lv()
print(lv)
times, states = lv.gillespie(2000)
out = smfsb.discretise(times, states, dt)

fig, axis = plt.subplots()
for i in range(2):
    axis.step(np.arange(0, times[-1], dt), out[:, i], where="post")

axis.legend(lv.n)
fig.savefig("s-m-lv-discretise.pdf")

# eof
