"""Cosmology setup"""

import numpy as np
from collections import namedtuple, OrderedDict

from jax import jit as jjit

from dsps.cosmology import flat_wcdm
from dsps.cosmology import DEFAULT_COSMOLOGY as DEFAULT_COSMOLOGY_DSPS

# default diffhalos cosmology
DEFAULT_COSMOLOGY = OrderedDict(
    Om0=0.3111,
    sigma8=0.8102,
    ns=0.9665,
    Ob0=0.0490,
    H0=67.66,
    w0=-1.0,
    wa=0.0,
)

DEFAULT_COSMO_NAMES = list(DEFAULT_COSMOLOGY.keys())
N_DEFAULT_COSMO_PARAMS = len(DEFAULT_COSMO_NAMES)

DEFAULT_COSMOLOGY_ARRAY = np.zeros(N_DEFAULT_COSMO_PARAMS)
for i, _param in enumerate(DEFAULT_COSMOLOGY):
    DEFAULT_COSMOLOGY_ARRAY[i] = DEFAULT_COSMOLOGY[_param]

DEFAULT_COSMOLOGY_NTUP = namedtuple("default_cosmo", DEFAULT_COSMO_NAMES)(
    *DEFAULT_COSMOLOGY_ARRAY
)

# cosmological parameter priors
# --parameters have to be in `DEFAULT_COSMOLOGY`
DEFAULT_COSMO_PRIORS = OrderedDict(
    Om0=(0.2, 0.5),
    sigma8=(0.6, 1.0),
    ns=(0.8, 1.1),
)


# default colossus cosmology (planck 2018)
DEFAULT_COSMOLOGY_COLOSSUS = OrderedDict(
    flat=True,
    Om0=0.3111,
    sigma8=0.8102,
    ns=0.9665,
    Ob0=0.0490,
    H0=67.66,
)

__all__ = ("get_tobs_from_zobs",)


@jjit
def get_tobs_from_zobs(z_obs, cosmo_params=DEFAULT_COSMOLOGY_DSPS):
    """
    Compute cosmic time at observation and at today,
    provided the corresponding redshifts
    and a cosmology

    Parameters
    ----------
    z_obs: ndarray of shape (n_t, )
        redshift values

    cosmo_params: namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    Returns
    -------
    t_obs: ndarray of shape (n_t, )
        cosmic time at observation, in Gyr

    t_0: float
        cosmic time at today, in Gyr
    """

    t_obs = flat_wcdm.age_at_z(z_obs, *cosmo_params)
    t_0 = flat_wcdm.age_at_z0(*cosmo_params)

    return t_obs, t_0
