#!/usr/bin/env python3

import smfsb
import numpy as np
import matplotlib.pyplot as plt
import smfsb.models

M=200
N=250
T=30
x0 = np.zeros((2,M,N))
lv = smfsb.models.lv()
x0[:,int(M/2),int(N/2)] = lv.m
stepLv2d = lv.stepCLE2D(np.array([0.6, 0.6]), 0.1)
x1 = stepLv2d(x0, 0, T)

fig, axis = plt.subplots()
for i in range(2):
    axis.imshow(x1[i,:,:])
    axis.set_title(lv.n[i])
    fig.savefig(f"stepCLE2Df{i}.pdf")


    
    
# eof
