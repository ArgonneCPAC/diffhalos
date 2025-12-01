""" """

import numpy as np
from jax import random as jran

from .. import stratified_grid as sg


def test_stratified_xy_grid():
    ran_key = jran.key(0)
    n_tests = 100
    for n_per_dim in (5, 50, 500):
        for __ in range(n_tests):
            ran_key, test_key = jran.split(ran_key, 2)
            xy_grid = sg.stratified_xy_grid(n_per_dim, test_key)
            assert xy_grid.shape == (n_per_dim**2, 2)
            assert np.all(xy_grid >= 0)
            assert np.all(xy_grid <= 1)

    xy_grid = sg.stratified_xy_grid(1_000, ran_key)
    assert np.allclose(xy_grid.mean(), 0.5, atol=0.1)
