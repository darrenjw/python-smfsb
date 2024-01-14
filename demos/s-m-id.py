#!/usr/bin/env python3

import smfsb
import smfsb.models

id = smfsb.models.id()
print(id)
step = id.stepGillespie()
out = smfsb.simTs(id.m, 0, 100, 0.1, step)

import matplotlib.pyplot as plt
fig, axis = plt.subplots()
axis.plot(range(out.shape[0]), out[:,0])

axis.legend(id.n)
fig.savefig("s-m-id.pdf")

# eof
