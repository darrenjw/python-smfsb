#!/usr/bin/env python3

# Lotka-Volterra model with a CLE approximation

import smfsb
import smfsb.models
import numpy as np
import matplotlib.pyplot as plt


lv = smfsb.models.lv()
print(lv)
step_lv = lv.step_cle()
out = smfsb.sim_time_series(np.random.default_rng(), lv.m, 0, 100, 0.1, step_lv)

fig, axis = plt.subplots()
for i in range(2):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(lv.n)
fig.savefig("s-m-lv-cle.pdf")

# eof
