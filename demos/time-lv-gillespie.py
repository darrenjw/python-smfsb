#!/usr/bin/env python3
# time-lv-gillespie.py
# time the gillespie algorithm

import numpy as np
import scipy as sp
import smfsb
import matplotlib.pyplot as plt
import time

lvmod = smfsb.models.lv()
step = lvmod.stepGillespie()

## Start timer
startTime = time.time()
out = smfsb.simSample(10000, lvmod.m, 0, 20, step)
endTime = time.time()
## End timer
elapsedTime = endTime - startTime
print(f"\n\nElapsed time: {elapsedTime} seconds\n\n")

out = np.where(out > 1000, 1000, out)
print(sp.stats.describe(out))
fig, axes = plt.subplots(2,1)
for i in range(2):
    axes[i].hist(out[:,i], bins=50)
fig.savefig("time-lv-gillespie.pdf")


# eof

