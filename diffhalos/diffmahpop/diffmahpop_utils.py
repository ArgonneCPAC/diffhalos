"""
Useful diffmahnet functions
See https://github.com/ArgonneCPAC/diffmah/tree/main/diffmah/diffmahpop_kernels
"""

import jax.numpy as jnp
from jax import jit as jjit
from jax import vmap

from diffmah.diffmahpop_kernels.mc_bimod_cens import mc_cenpop
from diffmah.diffmah_kernels import _log_mah_kern
from diffmah.diffmahpop_kernels.bimod_censat_params import (
    DEFAULT_DIFFMAHPOP_PARAMS,
)

from .utils import rescale_mah_parameters

T_GRID_MIN = 0.5
T_GRID_MAX = 13.8
N_T_GRID = 100

__all__ = ("mc_mah_cenpop",)

log_mah_kern_vmap = jjit(vmap(_log_mah_kern, in_axes=(0, None, None)))


def mc_mah_cenpop(
    m_obs,
    t_obs,
    randkey,
    logt0,
    n_sample=1,
    params=DEFAULT_DIFFMAHPOP_PARAMS,
    t_min=T_GRID_MIN,
    t_max=T_GRID_MAX,
    n_t=N_T_GRID,
    return_mah_params_and_der=False,
):
    """
    Diffmahpop predictions for populations of halo MAHs

    Parameters
    ----------
    m_obs: ndarray of shape (n_halo, )
        grid of base-10 log of mass of the halos at observation, in Msun

    t_obs: ndarray of shape (n_halo, )
        grid of base-10 log of cosmic time at observation of each halo, in Gyr

    randkey: key
        JAX random key

    logt0: float
        base-10 log of the age of the Universe at z=0, in Gyr

    n_sample: int
        number of MC samples per (m_obs,t_obs) pair

    params: namedtuple
        diffmah parameters

    t_min: float
        base-10 log of minimum value for time grid
        at which to compute mah, in Gyr

    t_max: float
        base-10 log of maximum value for time grid,
        at which to compute mah, in Gyr

    n_t: int
        number of points in time grid

    return_mah_params_and_der: bool
        if True the MAH parameters and MAH gradients
        will also be returned

    Returns
    -------
    log_mah: ndarray of shape (n_halo, n_t)
        base-10 log of mah for each halo, in Msun

    t_grid: ndarray of shape (n_t, )
        cosmic time grid on which to compute MAHs, in Gyr

    mah_params: namedtuple of ndarrays of shape (n_halo,)
        mah parameters for all halos in the population
    """
    # get a list of (m_obs, t_obs) for each MC realization
    m_vals, t_vals = [
        jnp.repeat(x.flatten(), n_sample)
        for x in jnp.meshgrid(
            m_obs,
            t_obs,
        )
    ]

    # construct time grids for each halo, given observation time
    t_grid = jnp.linspace(t_min, t_max, n_t)

    # predict uncorrected MAHs
    mah_params_uncorrected, _, log_mah_uncorrected = mc_cenpop(
        params,
        t_grid,
        m_vals,
        t_vals,
        randkey,
        logt0,
    )

    # rescale the mah parameters to the correct logm0
    mah_params = rescale_mah_parameters(
        mah_params_uncorrected,
        m_vals,
        log_mah_uncorrected[:, -1],
    )

    # get the corrected MAHs
    log_mah = log_mah_kern_vmap(mah_params, t_grid, logt0)

    if return_mah_params_and_der:
        return log_mah, t_grid, mah_params

    return log_mah, t_grid
