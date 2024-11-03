# test_data.py
# tests relating to the synthetic data sets

import smfsb


def test_lvperfect():
    assert smfsb.data.lv_perfect.shape == (16, 3)


def test_lvnoise10():
    assert smfsb.data.lv_noise_10.shape == (16, 3)


def test_lvpreynoise10():
    assert smfsb.data.lv_prey_noise_10.shape == (16, 2)


# eof
