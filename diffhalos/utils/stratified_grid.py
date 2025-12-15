""" """

from functools import partial

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

__all__ = ("stratified_xy_grid",)


@partial(jjit, static_argnames=["n_per_dim"])
def stratified_xy_grid(n_per_dim, ran_key):
    """
    Stratified grid with noise

    Parameters
    ----------
    n_per_dim: int
        number of points per dimension (total = n_per_dim^2)

    ran_key: jax.random.key(seed)
        random key

    Returns
    -------
    xy_grid: ndarray of shape (n_per_dim^2, 2)
        0 <= x,y <= 1 for every grid element
    """
    grid_1d = (jnp.arange(n_per_dim) + 0.5) / n_per_dim

    x = jnp.repeat(grid_1d, n_per_dim)
    y = jnp.tile(grid_1d, n_per_dim)
    xy_grid = jnp.column_stack([x, y])

    uran = jran.uniform(ran_key, shape=xy_grid.shape)
    noise = (uran - 0.5) / n_per_dim

    return xy_grid + noise


@partial(jjit, static_argnames=["n_per_dim"])
def stratified_grid_scaled(n_per_dim, ran_key, zmin, zmax, mmin, mmax):
    """
    Stratified grid with noise

    Parameters
    ----------
    n_per_dim: int
        number of points per dimension (total = n_per_dim^2)

    ran_key: jax.random.key(seed)
        random key

    zmin: float
        minimum redshift

    zmax: float
        minimum redshift

    mmin: float
        base-10 log of minimum mass, in Msun

    mmax: float
        base-10 log of maximum mass, in Msun

    Returns
    -------
    z_grid: ndarray of shape (n_per_dim^2, )
        redshift grid points

    m_grid: ndarray of shape (n_per_dim^2, )
        base-10 log of mass grid points, in Msun
    """
    xy_grid = stratified_xy_grid(n_per_dim, ran_key)
    z_grid = map_intervals(xy_grid[:, 0], 0, 1, zmin, zmax)
    m_grid = map_intervals(xy_grid[:, 1], 0, 1, mmin, mmax)

    return z_grid, m_grid


@jjit
def map_intervals(values, oldMin, oldMax, newMin, newMax):
    """
    Helper function to map values in the interval [oldMin, oldMax]
    to the new interval [newMin, newMax]
    """
    return ((values - oldMin) / (oldMax - oldMin)) * (newMax - newMin) + newMin
