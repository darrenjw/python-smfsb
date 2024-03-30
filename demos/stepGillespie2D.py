#!/usr/bin/env python3

import smfsb
import numpy as np
import matplotlib.pyplot as plt
import smfsb.models

M=20
N=30
T=10
x0 = np.zeros((2,M,N))
lv = smfsb.models.lv()
x0[:,int(M/2),int(N/2)] = lv.m
stepLv2d = lv.stepGillespie2D(np.array([0.6, 0.6]))
x1 = stepLv2d(x0, 0, T)

fig, axis = plt.subplots()
for i in range(2):
    axis.imshow(x1[i,:,:])
    axis.set_title(lv.n[i])
    fig.savefig(f"stepGillespie2D{i}.pdf")


# eof
