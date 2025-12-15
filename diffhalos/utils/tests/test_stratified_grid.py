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


def test_stratified_grid_scaled():
    ran_key = jran.key(0)
    n_tests = 100

    zmin, zmax = 1.0, 4.5
    mmin, mmax = 11.5, 14.0

    for n_per_dim in (5, 50, 500):
        for __ in range(n_tests):
            ran_key, test_key = jran.split(ran_key, 2)
            z_grid, m_grid = sg.stratified_grid_scaled(
                n_per_dim, test_key, zmin, zmax, mmin, mmax
            )
            assert z_grid.size == n_per_dim**2
            assert m_grid.size == n_per_dim**2
            assert np.all((z_grid >= zmin) * (z_grid <= zmax))
            assert np.all((m_grid >= mmin) * (m_grid <= mmax))
