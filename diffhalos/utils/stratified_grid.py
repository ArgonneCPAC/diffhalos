""" """

from functools import partial

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran


@partial(jjit, static_argnames=["n_per_dim"])
def stratified_xy_grid(n_per_dim, ran_key):
    """
    Stratified grid with noise

    Parameters
    ----------
    n_per_dim: int
        Number of points per dimension (total = n_per_dim^2)

    ran_key: jax.random.key(seed)

    Returns
    -------
    xy_grid : array, shape (n_per_dim^2, 2)
        0 <= x,y <= 1 for every grid element

    """
    grid_1d = (jnp.arange(n_per_dim) + 0.5) / n_per_dim

    x = jnp.repeat(grid_1d, n_per_dim)
    y = jnp.tile(grid_1d, n_per_dim)
    xy_grid = jnp.column_stack([x, y])

    uran = jran.uniform(ran_key, shape=xy_grid.shape)
    noise = uran / n_per_dim - 0.5 / n_per_dim

    return xy_grid + noise
