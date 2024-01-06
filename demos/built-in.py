#!/usr/bin/env python3

import smfsb

print(smfsb.lv)
stepLv = smfsb.lv.stepGillespie()
out = smfsb.simTs(smfsb.lv.m, 0, 100, 0.1, stepLv)

import matplotlib.pyplot as plt
fig, axis = plt.subplots()
for i in range(2):
	axis.plot(range(out.shape[0]), out[:,i])

axis.legend(smfsb.lv.n)
fig.savefig("lv.pdf")

# eof
