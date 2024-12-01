#!/usr/bin/env python3

import smfsb
import smfsb.models
import matplotlib.pyplot as plt


bd = smfsb.models.bd()
print(bd)
step = bd.step_gillespie()
out = smfsb.sim_time_series(bd.m, 0, 100, 0.1, step)

fig, axis = plt.subplots()
axis.plot(range(out.shape[0]), out[:, 0])

axis.legend(bd.n)
fig.savefig("s-m-bd.pdf")

# eof
