#!/usr/bin/env python3

import smfsb
import smfsb.models
import matplotlib.pyplot as plt


dimer = smfsb.models.dimer()
print(dimer)
step = dimer.step_gillespie()
out = smfsb.sim_time_series(dimer.m, 0, 100, 0.1, step)

fig, axis = plt.subplots()
for i in range(2):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(dimer.n)
fig.savefig("s-m-dimer.pdf")

# eof
