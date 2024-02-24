#!/usr/bin/env python3

import smfsb
import numpy as np

seirSH = """
@model:3.1.1=SEIR "SEIR Epidemic model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:S=100 s
 Pop:E=0 s
 Pop:I=5 s
 Pop:R=0 s
@reactions
@r=Infection
 S + I -> E + I
 beta*S*I : beta=0.1
@r=Transition
 E -> I
 sigma*E : sigma=0.2
@r=Removal
 I -> R
 gamma*I : gamma=0.5
"""

seir = smfsb.sh2Spn(seirSH)
stepSeir = seir.stepGillespie()
out = smfsb.simTs(seir.m, 0, 40, 0.05, stepSeir)

import matplotlib.pyplot as plt
fig, axis = plt.subplots()
for i in range(len(seir.m)):
	axis.plot(np.arange(0, 40, 0.05), out[:,i])

axis.legend(seir.n)
fig.savefig("sbmlsh.pdf")

# eof

