"""Useful utilities to be used with the lightcone module"""

from jax import jit as jjit

from .cosmo_dsps import flat_wcdm
from .cosmo_jax import DEFAULT_COSMOLOGY_JAXCOSMO
from .cosmo_conversion import jaxcosmo_to_dsps_cosmology

__all__ = ("get_tobs_from_zobs",)


@jjit
def get_tobs_from_zobs(z_obs, cosmo_params=DEFAULT_COSMOLOGY_JAXCOSMO):
    """
    Compute cosmic time at observation and at today,
    provided the corresponding redshifts
    and a cosmology

    Parameters
    ----------
    z_obs: ndarray of shape (n_t, )
        redshift values

    cosmo_params: jax-cosmo parameters object
        cosmological parameters

    Returns
    -------
    t_obs: ndarray of shape (n_t, )
        cosmic time at observation, in Gyr

    t_0: float
        cosmic time at today, in Gyr
    """

    cosmo_params = jaxcosmo_to_dsps_cosmology(cosmo_params)
    t_obs = flat_wcdm.age_at_z(z_obs, *cosmo_params)
    t_0 = flat_wcdm.age_at_z0(*cosmo_params)

    return t_obs, t_0
