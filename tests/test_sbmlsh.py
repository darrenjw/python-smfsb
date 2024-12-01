# test_sbmlsh
# tests relating to SBML and SBML-shorthand

import smfsb


seir_sh = """
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
    seir = smfsb.shorthand_to_spn(seir_sh)
    step_seir = seir.step_gillespie()
    out = smfsb.sim_time_series(seir.m, 0, 40, 0.05, step_seir)
    assert out.shape == (800, 4)
    assert out[400, 3] >= 0


# eof
