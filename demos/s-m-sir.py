#!/usr/bin/env python3

import smfsb
import smfsb.models

sir = smfsb.models.sir()
print(sir)
step = sir.stepGillespie()
out = smfsb.simTs(sir.m, 0, 100, 0.1, step)

import matplotlib.pyplot as plt
fig, axis = plt.subplots()
for i in range(3):
	axis.plot(range(out.shape[0]), out[:,i])

axis.legend(sir.n)
fig.savefig("s-m-sir.pdf")

# eof
