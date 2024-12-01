#!/usr/bin/env python3

import smfsb
import smfsb.models
import matplotlib.pyplot as plt

mm = smfsb.models.mm()
print(mm)
step = mm.step_gillespie()
out = smfsb.sim_time_series(mm.m, 0, 100, 0.1, step)


fig, axis = plt.subplots()
for i in range(4):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(mm.n)
fig.savefig("s-m-mm.pdf")

# eof
