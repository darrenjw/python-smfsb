#!/usr/bin/env python3

import smfsb
import numpy as np
import matplotlib.pyplot as plt
import smfsb.models

N = 20
T = 30
x0 = np.zeros((2, N))
lv = smfsb.models.lv()
x0[:, int(N / 2)] = lv.m
step_lv1d = lv.step_gillespie_1d(np.array([0.6, 0.6]))
x1 = step_lv1d(x0, 0, 1)
print(x1)
out = smfsb.sim_time_series_1d(x0, 0, T, 1, step_lv1d, True)
# print(out)

fig, axis = plt.subplots()
for i in range(2):
    axis.imshow(out[i, :, :])
    axis.set_title(lv.n[i])
    fig.savefig(f"step_gillespie_1d{i}.pdf")


# eof
