#!/usr/bin/env python3

import smfsb
import smfsb.models

lv = smfsb.models.lv()
print(lv)
stepLv = lv.step_first()
out = smfsb.sim_time_series(lv.m, 0, 100, 0.1, stepLv)

import matplotlib.pyplot as plt

fig, axis = plt.subplots()
for i in range(2):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(lv.n)
fig.savefig("s-m-lv-frm.pdf")

# eof
