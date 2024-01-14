#!/usr/bin/env python3

import smfsb
import smfsb.models

mm = smfsb.models.mm()
print(mm)
step = mm.stepGillespie()
out = smfsb.simTs(mm.m, 0, 100, 0.1, step)

import matplotlib.pyplot as plt
fig, axis = plt.subplots()
for i in range(4):
	axis.plot(range(out.shape[0]), out[:,i])

axis.legend(mm.n)
fig.savefig("s-m-mm.pdf")

# eof
