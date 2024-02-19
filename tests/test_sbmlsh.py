# test_sbmlsh
# tests relating to SBML and SBML-shorthand

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

def test_sbmlsh():
    seir = smfsb.sh2Spn(seirSH)
    stepSeir = seir.stepGillespie()
    out = smfsb.simTs(seir.m, 0, 40, 0.05, stepSeir)
    assert(out.shape == (800, 4))
    assert(out[400,3] >= 0)



# eof

