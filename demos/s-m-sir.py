#!/usr/bin/env python3

import smfsb
import smfsb.models
import matplotlib.pyplot as plt

sir = smfsb.models.sir()
print(sir)
step = sir.step_gillespie()
out = smfsb.sim_time_series(sir.m, 0, 100, 0.1, step)


fig, axis = plt.subplots()
for i in range(3):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(sir.n)
fig.savefig("s-m-sir.pdf")

# eof
