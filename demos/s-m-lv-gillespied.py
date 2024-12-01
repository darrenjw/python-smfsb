#!/usr/bin/env python3

import smfsb
import smfsb.models
import numpy as np
import matplotlib.pyplot as plt


lv = smfsb.models.lv()
print(lv)
states = lv.gillespied(30, 0.1)
print(states.shape)


fig, axis = plt.subplots()
for i in range(2):
    axis.step(np.arange(0, 30, 0.1), states[:, i], where="post")

axis.legend(lv.n)
fig.savefig("s-m-lv-gillespied.pdf")

# eof
