"""Useful utilities to be used with the lightcone module"""

from collections import namedtuple
from functools import partial

from jax import numpy as jnp
from jax import jit as jjit

from ..cosmology import DEFAULT_COSMOLOGY
from ..cosmology.cosmo_basics import get_tobs_from_zobs

__all__ = ("generate_mock_cenpop",)


@partial(jjit, static_argnames=["n_cens"])
def generate_mock_cenpop(
    z_min,
    z_max,
    logmp_min,
    logmp_max,
    cosmo_params=DEFAULT_COSMOLOGY,
    n_cens=200,
):
    """
    Convenience function to easily get a mock central
    halo population between two redshifts and masses,
    on a regular grid

    Parameters
    ----------
    z_min: float
        minumum redshift

    z_max: float
        maximum redshift

    logmp_min: float
        minumum halo mass, in Msun

    logmp_max: float
        minumum halo mass, in Msun

    cosmo_params: namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    n_cens: int
        number of generated halos

    Returns
    -------
    cenpop: namedtuple
        central halo population with fields
            logmp_obs: ndarray of shape (n_host, )
                base-10 log of halo mass at observation, in Msun

            t_obs: ndarray of shape (n_host, )
                cosmic time at observation, in Gyr

            logt0: float
                base-10 log of cosmic time at today, in Gyr
    """
    z_obs = jnp.linspace(z_min, z_max, n_cens)
    logmp_obs = jnp.linspace(logmp_min, logmp_max, n_cens)

    t_obs, t_0 = get_tobs_from_zobs(z_obs, cosmo_params=cosmo_params)
    logt0 = jnp.log10(t_0)

    fields = ("logmp_obs", "t_obs", "logt0")
    values = (logmp_obs, t_obs, logt0)

    cenpop = namedtuple("cenpop", fields)(*values)

    return cenpop
