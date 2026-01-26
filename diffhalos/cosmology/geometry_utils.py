"""Utility functions for lightcone calculations"""

from dsps.cosmology import flat_wcdm
from jax import grad
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..defaults import FULL_SKY_AREA

_Z = (0, None, None, None, None)
d_Rcom_dz_func = jjit(
    vmap(grad(flat_wcdm.comoving_distance_to_z, argnums=0), in_axes=_Z)
)

__all__ = (
    "spherical_shell_comoving_volume",
    "compute_volume_from_sky_area",
)


@jjit
def spherical_shell_comoving_volume(z_grid, cosmo_params):
    """
    Comoving volume of a spherical shell with width ΔR

    Parameters
    ----------
    z_grid: ndarray of shape (n_z, )
        grid of redshift values

    cosmo_params: namedtuple
        cosmological parameters

    Returns
    -------
    vol_shell_grid: float
        volume of grid shell, in comoving Mpc**3
    """

    # Compute comoving distance to each grid point
    r_grid = flat_wcdm.comoving_distance(z_grid, *cosmo_params)

    # Compute ΔR = (∂R/∂z)*Δz
    d_r_grid_dz = d_Rcom_dz_func(z_grid, *cosmo_params)
    d_z_grid = z_grid[1] - z_grid[0]
    d_r_grid = d_r_grid_dz * d_z_grid

    # vol_shell_grid = 4π*R*R*ΔR
    vol_shell_grid = 4 * jnp.pi * r_grid * r_grid * d_r_grid

    return vol_shell_grid


@jjit
def compute_volume_from_sky_area(
    redshift,
    sky_area_degsq,
    cosmo_params,
):
    """
    Helper function to compute the comoving volume
    at given redshift from the given sky area

    Parameters
    ----------
    redshift: ndarray of shape (n_z, )
        redshift value

    sky_area_degsq: float
        sky area, in deg^2

    cosmo_params: namedtuple
        cosmological paramters

    Returns
    -------
    vol_shell_grid_mpc: ndarray of shape (n_z, )
        comoving volume, in Mpc^3
    """
    redshift = jnp.atleast_1d(redshift)

    fsky = sky_area_degsq / FULL_SKY_AREA
    vol_shell_grid_mpc = fsky * spherical_shell_comoving_volume(
        redshift,
        cosmo_params,
    )

    return vol_shell_grid_mpc
