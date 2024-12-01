#!/usr/bin/env python3

import smfsb
import smfsb.models
import matplotlib.pyplot as plt

lv = smfsb.models.lv([0.2, 0.001, 0.1])
print(lv)
step_lv = lv.step_gillespie()
out = smfsb.sim_time_series(lv.m, 0, 100, 0.1, step_lv)


fig, axis = plt.subplots()
for i in range(2):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(lv.n)
fig.savefig("s-m-lv2.pdf")

# eof
