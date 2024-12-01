#!/usr/bin/env python3
# time-lv-cle.py
# time the CLE algorithm

import numpy as np
import scipy as sp
import smfsb
import matplotlib.pyplot as plt
import time

lvmod = smfsb.models.lv()
step = lvmod.step_cle(0.01)

## Start timer
start_time = time.time()
out = smfsb.sim_sample(10000, lvmod.m, 0, 20, step)
end_time = time.time()
## End timer
elapsed_time = end_time - start_time
print(f"\n\nElapsed time: {elapsed_time} seconds\n\n")

out = np.where(out > 1000, 1000, out)
print(sp.stats.describe(out))
fig, axes = plt.subplots(2, 1)
for i in range(2):
    axes[i].hist(out[:, i], bins=50)
fig.savefig("time-lv-cle.pdf")


# eof
