#!/usr/bin/env python3

import smfsb
import smfsb.models

bd = smfsb.models.bd()
print(bd)
step = bd.stepGillespie()
out = smfsb.simTs(bd.m, 0, 100, 0.1, step)

import matplotlib.pyplot as plt
fig, axis = plt.subplots()
axis.plot(range(out.shape[0]), out[:,0])

axis.legend(bd.n)
fig.savefig("s-m-bd.pdf")

# eof
