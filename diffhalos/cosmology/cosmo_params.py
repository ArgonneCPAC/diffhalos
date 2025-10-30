"""Cosmological parameters"""

import typing

from jax import numpy as jnp


class CosmoParams(typing.NamedTuple):
    """
    This handles cosmology like in DSPS and
    it is a convenient way of storing cosmologies

    Parameters
    ----------
    Om0: float
        matter density parameter today

    w0: float
        dark energy equation of state parameter

    wa: float
        dark energy equation of state redshift-dependence parameter

    h: float
        Hubble constant in units of 100 km/Mpc/s

    Returns
    -------
    namedtuple storing parameters of a flat w0-wa cdm cosmology
    """

    Om0: jnp.float32
    w0: jnp.float32
    wa: jnp.float32
    h: jnp.float32


#########################
# Define some cosmologies
#########################

PLANCK15 = CosmoParams(0.3075, -1.0, 0.0, 0.6774)
WMAP5 = CosmoParams(0.277, -1.0, 0.0, 0.702)
FSPS_COSMO = CosmoParams(0.27, -1.0, 0.0, 0.72)
COSMOS20 = CosmoParams(0.3, -1.0, 0.0, 0.7)
