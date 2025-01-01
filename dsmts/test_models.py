# test_models.py
# Tests to run using pytest

import dsmts
import numpy as np

N = 100


def check_model(file_stem):
    fails = dsmts.test_model(N, file_stem)
    assert np.sum(fails) == 0


# List of models to check:


def test_m0001():
    check_model("stochastic/00001/dsmts-001-01")


def test_m0020():
    check_model("stochastic/00020/dsmts-002-01")


def test_m0030():
    check_model("stochastic/00030/dsmts-003-01")


# eof
