#!/usr/bin/env python3

import smfsb
import smfsb.models
import matplotlib.pyplot as plt

lv = smfsb.models.lv()
print(lv)
times, states = lv.gillespie(2000)


fig, axis = plt.subplots()
for i in range(2):
    axis.step(times, states[1:, i], where="post")

axis.legend(lv.n)
fig.savefig("s-m-lv-gillespie.pdf")

# eof
