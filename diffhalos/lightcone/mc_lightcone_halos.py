# flake8: noqa: E402
"""Functions to generate host halo lightcones"""

from jax import config

config.update("jax_enable_x64", True)

from functools import partial

from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from diffmah.diffmah_kernels import _log_mah_kern

from ..cosmology import flat_wcdm, DEFAULT_COSMOLOGY
from ..cosmology.cosmo_basics import get_tobs_from_zobs
from ..hmf.hmf_model import halo_lightcone_weights
from ..hmf import mc_hosts
from ..mah.utils import apply_mah_rescaling
from ..cosmology.geometry_utils import compute_volume_from_sky_area

N_HMF_GRID = 2_000
DEFAULT_LOGMP_CUTOFF = 10.0
DEFAULT_LOGMP_HIMASS_CUTOFF = 14.5

DEFAULT_DIFFMAHNET_CEN_MODEL = "cenflow_v2_0_64bit.eqx"

_AXES = (0, None, None, 0, None)
mc_logmp_vmap = jjit(vmap(mc_hosts._mc_host_halos_singlez_kern, in_axes=_AXES))

__all__ = ("mc_lc_hmf", "mc_lc_halos", "weighted_lc_halos")


def mc_lc_hmf(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    cosmo_params=DEFAULT_COSMOLOGY,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    lgmp_max=mc_hosts.LGMH_MAX,
    n_hmf_grid=N_HMF_GRID,
):
    """
    Generate a Monte Carlo realization of a lightcone of
    host halo masses and redshifts, between two redshifts
    and between a minimum and a maximum mass

    Parameters
    ----------
    ran_key: jran.key
        random key

    lgmp_min: float
        minimum halo mass, in Msun

    z_min: float
        minimum redshift

    z_max: float
        maximum redshift

    sky_area_degsq: float
        sky area, in deg^2

    cosmo_params: namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    hmf_params: namedtuple
        halo mass function parameters

    lgmp_max: float
        base-10 log of maximum halo mass, in Msun

    n_hmf_grid: int
        number of redshift grid points for HMF computations

    Returns
    -------
    z_cenpop: ndarray of shape (n_halos, )
        redshifts distributed randomly within the lightcone volume

    logmp_cenpop: ndarray of shape (n_halos, )
        halo masses derived by Monte Carlo sampling the halo mass function
        at the appropriate redshift for each point
    """

    # randoms for Nhalos, halo mass, and redshift
    halo_counts_key, m_key, z_key = jran.split(ran_key, 3)

    # set up a uniform grid in redshift
    z_grid = jnp.linspace(z_min, z_max, n_hmf_grid)

    # compute the comoving volume of a thin shell at each grid point
    volume_com_mpc = compute_volume_from_sky_area(
        z_grid,
        sky_area_degsq,
        cosmo_params,
    )

    # at each grid point, compute <Nhalos> for the shell volume
    mean_nhalos = mc_hosts._compute_nhalos_tot(
        hmf_params,
        lgmp_min,
        z_grid,
        volume_com_mpc,
    )
    mean_nhalos_lgmax = mc_hosts._compute_nhalos_tot(
        hmf_params,
        lgmp_max,
        z_grid,
        volume_com_mpc,
    )
    mean_nhalos = mean_nhalos - mean_nhalos_lgmax

    # at each grid point, compute a Poisson realization of <Nhalos>
    nhalos_tot = jran.poisson(halo_counts_key, mean_nhalos).sum()

    # compute the CDF of the volume
    weights_grid = mean_nhalos / mean_nhalos.sum()
    cdf_grid = jnp.cumsum(weights_grid)

    # assign redshift via inverse transformation sampling of the halo counts CDF
    uran_z = jran.uniform(z_key, minval=0, maxval=1, shape=(nhalos_tot,))
    z_cenpop = jnp.interp(uran_z, cdf_grid, z_grid)

    # randoms used in inverse transformation sampling halo mass
    uran_m = jran.uniform(m_key, minval=0, maxval=1, shape=(nhalos_tot,))

    # draw a halo mass from the HMF at the particular redshift of each halo
    logmp_cenpop = mc_logmp_vmap(uran_m, hmf_params, lgmp_min, z_cenpop, lgmp_max)

    return z_cenpop, logmp_cenpop


def mc_lc_halos(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    cosmo_params=DEFAULT_COSMOLOGY,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
    lgmp_max=mc_hosts.LGMH_MAX,
    n_hmf_grid=N_HMF_GRID,
    centrals_model_key=DEFAULT_DIFFMAHNET_CEN_MODEL,
):
    """
    Generate a halo lightcone, including MAHs,
    between a minimum and a maximum value of redshift and halo mass

    Parameters
    ----------
    ran_key: jran.key
        random key

    lgmp_min: float
        minimum halo mass, in Msun

    z_min: float
        minimum redshift

    z_max: float
        maximum redshift

    sky_area_degsq: float
        sky area, in deg^2

    nhalos_tot: int
        total number of halos to generate in the lightcone

    cosmo_params: namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    hmf_params: namedtuple
        halo mass function parameters

    logmp_cutoff: float
        base-10 log of minimum halo mass for which
        diffmahnet is used to generate MAHs, in Msun;
        for logmp < logmp_cutoff, P(θ_MAH | logmp) = P(θ_MAH | logmp_cutoff)

    logmp_cutoff_himass: float
        base-10 log of maximum halo mass for which
        diffmahnet is used to generate MAHs, in Msun

    lgmp_max: float
        base-10 log of maximum host halo mass, in Msun

    n_hmf_grid: int
        number of redshift grid points for HMF computations

    centrals_model_key: str
        diffmahnet model to use for centrals

    Returns
    -------
    cenpop: namedtuple
        host halo population with fields:
            z_obs: ndarray of shape (n_halos, )
                lightcone redshift

            logmp_obs: ndarray of shape (n_halos, )
                halo mass at the lightcone redshift, in Msun

            mah_params: namedtuple of ndarray's with shape (n_halos, n_mah_params)
                diffmah parameters for each host halo in the lightcone

            logmp0: narray of shape (n_halos, )
                base-10 log of halo mass at z=0, in Msun

            logt0: float
                base-10 log of cosmic time at today, in Gyr
    """

    # generate mc realization of the halo mass function
    lc_hmf_key, mah_key = jran.split(ran_key, 2)
    z_obs, logmp_obs_mf = mc_lc_hmf(
        lc_hmf_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
        cosmo_params=cosmo_params,
        hmf_params=hmf_params,
        lgmp_max=lgmp_max,
        n_hmf_grid=n_hmf_grid,
    )

    t_obs, t_0 = get_tobs_from_zobs(z_obs, cosmo_params=cosmo_params)
    logt0 = jnp.log10(t_0)

    # get rescaled mah parameters and mah's
    logmp_obs_clipped = jnp.clip(logmp_obs_mf, logmp_cutoff, logmp_cutoff_himass)
    mah_params, logmp_obs = apply_mah_rescaling(
        mah_key,
        logmp_obs_mf,
        logmp_obs_clipped,
        t_obs,
        logt0,
        centrals_model_key,
    )

    # compute MAH values today
    logmp0 = _log_mah_kern(mah_params, 10**logt0, logt0)

    # create output namedtuple
    fields = ("z_obs", "t_obs", "logmp_obs", "mah_params", "logmp0", "logt0")
    values = (z_obs, t_obs, logmp_obs, mah_params, logmp0, logt0)
    cenpop = namedtuple("cenpop", fields)(*values)

    return cenpop


@partial(jjit, static_argnames=["centrals_model_key"])
def weighted_lc_halos(
    ran_key,
    z_obs,
    logmp_obs,
    sky_area_degsq,
    cosmo_params=DEFAULT_COSMOLOGY,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
    centrals_model_key=DEFAULT_DIFFMAHNET_CEN_MODEL,
):
    """
    Generates a weighted lightcone population of halos with MAHs,
    on an input grid of redshift and mass

    Parameters
    ----------
    ran_key: jran.key
        random key

    z_obs: ndarray of shape (n_halo, )
        observed redshifts of galaxies

    logmp_obs: ndarray of shape (n_halo, )
        base-10 log of observed halo masses, in Msun

    sky_area_degsq: float
        sky area, in deg^2

    cosmo_params: namedtuple
        cosmological parameters

    hmf_params: namedtuple
        halo mass function parameters

    logmp_cutoff: float
        base-10 log of minimum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun;
        for logmp < logmp_cutoff, P(θ_MAH | logmp) = P(θ_MAH | logmp_cutoff)

    logmp_cutoff_himass: float
        base-10 log of maximum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun

    centrals_model_key: str
        diffmahnet model to use for centrals

    Returns
    -------
    cenpop: namedtuple
        host halo population with fields:
            z_obs: ndarray of shape (n_halo, )
                redshift values

            t_obs: ndarray of shape (n_halo, )
                cosmic time at observation, in Gyr

            logmp_obs: ndarray of shape (n_halo, )
                base-10 log of halo mass at observation, in Msun

            mah_params: namedtuple of ndarrays of shape (n_halo, )
                mah parameters

            logmp0: ndarray of shape (n_halo, )
                base-10 log of halo mass at z=0, in Msun

            nhalos: ndarray of shape (n_halo, )
                weighted number of halos at each grid point
    """
    # get halo weights
    nhalo_weights = halo_lightcone_weights(
        logmp_obs,
        z_obs,
        sky_area_degsq,
        hmf_params=hmf_params,
        cosmo_params=cosmo_params,
    )

    t_obs, t_0 = get_tobs_from_zobs(z_obs, cosmo_params=cosmo_params)
    logt0 = jnp.log10(t_0)

    # get rescaled mah parameters and mah's
    logmp_obs_clipped = jnp.clip(logmp_obs, logmp_cutoff, logmp_cutoff_himass)
    mah_params, logmp_obs = apply_mah_rescaling(
        ran_key,
        logmp_obs,
        logmp_obs_clipped,
        t_obs,
        logt0,
        centrals_model_key,
    )

    # compute MAH values today
    logmp0 = _log_mah_kern(mah_params, 10**logt0, logt0)

    # create output namedtuple
    fields = ("z_obs", "t_obs", "logmp_obs", "mah_params", "logmp0", "logt0", "nhalos")
    values = (z_obs, t_obs, logmp_obs, mah_params, logmp0, logt0, nhalo_weights)
    cenpop = namedtuple("cenpop", fields)(*values)

    return cenpop
