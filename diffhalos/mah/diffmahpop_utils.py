"""
Useful diffmahnet functions
See https://github.com/ArgonneCPAC/diffmah/tree/main/diffmah/diffmahpop_kernels
"""

import jax.numpy as jnp
from jax import jit as jjit
from jax import vmap

from diffmah.diffmahpop_kernels.mc_bimod_cens import mc_cenpop
from diffmah.diffmahpop_kernels.mc_bimod_sats import mc_satpop
from diffmah.diffmah_kernels import _log_mah_kern
from diffmah.diffmahpop_kernels.bimod_censat_params import (
    DEFAULT_DIFFMAHPOP_PARAMS,
)

from .utils import rescale_mah_parameters

LOGT0 = jnp.log10(13.8)

__all__ = (
    "mc_mah_cenpop",
    "mc_mah_satpop",
)

log_mah_kern_vmap = jjit(vmap(_log_mah_kern, in_axes=(0, None, None)))


def mc_mah_cenpop(
    m_obs,
    t_obs,
    ran_key,
    t_grid,
    params=DEFAULT_DIFFMAHPOP_PARAMS,
    logt0=LOGT0,
):
    """
    Generate populations of central halo MAHs using``diffmahpop``.
    This function gnerates MAH parameters that are 'corrected', so that
    logm0 is rescaled to match the true observed mass at time of observation.

    Parameters
    ----------
    m_obs: ndarray of shape (n_cens, )
        grid of base-10 log of mass of the halos at observation, in Msun

    t_obs: ndarray of shape (n_cens, )
        grid of base-10 log of cosmic time at observation of each halo, in Gyr

    ran_key: key
        JAX random key

    t_grid: ndarray of shape (n_cens, n_t)
        cosmic time in time grid for each negerated MAH, in Gyr

    params: namedtuple
        diffmahpop parameters for centrals

    logt0: float
        base-10 log of cosmic time today, in Gyr

    Returns
    -------
    cen_mah: ndarray of shape (n_cens, n_t)
        base-10 log of halo mass assembly histories,
        for all MC realizations, in Msun

    mah_params_corrected: namedtuple
        diffmah parameters from normalizing flow,
        each parameter is a ndarray of shape(n_cens, )
    """

    # predict uncorrected MAHs
    mah_params_uncorrected, _, logm_obs_uncorrected = mc_cenpop(
        params,
        t_grid,
        m_obs,
        t_obs,
        ran_key,
        logt0,
    )

    # rescale the mah parameters to the correct logm0
    mah_params_corrected = rescale_mah_parameters(
        mah_params_uncorrected,
        m_obs,
        logm_obs_uncorrected[:, -1],
    )

    # get the corrected MAHs
    cen_mah = log_mah_kern_vmap(mah_params_corrected, t_grid, logt0)

    return cen_mah, mah_params_corrected


def mc_mah_satpop(
    m_obs,
    t_obs,
    ran_key,
    t_grid,
    params=DEFAULT_DIFFMAHPOP_PARAMS,
    logt0=LOGT0,
):
    """
    Generate populations of subhalo MAHs using``diffmahpop``.
    This function generates MAH parameters that are 'corrected', so that
    logm0 is rescaled to match the true observed mass at time of observation.

    Parameters
    ----------
    m_obs: ndarray of shape (n_cens, )
        grid of base-10 log of mass of the halos at observation, in Msun

    t_obs: ndarray of shape (n_cens, )
        grid of base-10 log of cosmic time at observation of each halo, in Gyr

    ran_key: key
        JAX random key

    t_grid: ndarray of shape (n_cens, n_t)
        cosmic time in time grid for each negerated MAH, in Gyr

    params: namedtuple
        diffmahpop parameters for satellites

    logt0: float
        base-10 log of cosmic time today, in Gyr

    Returns
    -------
    sat_mah: ndarray of shape (n_subs, n_t)
        base-10 log of halo mass assembly histories,
        for all MC realizations, in Msun

    mah_params_corrected: namedtuple
        diffmah parameters from normalizing flow,
        each parameter is a ndarray of shape(n_subs, )
    """

    # predict uncorrected MAHs
    mah_params_uncorrected, _, logm_obs_uncorrected = mc_satpop(
        params,
        t_grid,
        m_obs,
        t_obs,
        ran_key,
        logt0,
    )

    # rescale the mah parameters to the correct logm0
    mah_params_corrected = rescale_mah_parameters(
        mah_params_uncorrected,
        m_obs,
        logm_obs_uncorrected[:, -1],
    )

    # get the corrected MAHs
    sat_mah = log_mah_kern_vmap(mah_params_corrected, t_grid, logt0)

    return sat_mah, mah_params_corrected
