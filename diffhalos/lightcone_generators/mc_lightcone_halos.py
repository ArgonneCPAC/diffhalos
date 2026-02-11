# flake8: noqa: E402
"""Functions to generate host halo lightcones"""

from jax import config

config.update("jax_enable_x64", True)

from collections import namedtuple
from functools import partial

from diffmah.diffmah_kernels import _log_mah_kern
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..cosmology.cosmo_jax import DEFAULT_COSMOLOGY_JAXCOSMO as DEFAULT_COSMOLOGY
from ..cosmology.cosmo_basics import get_tobs_from_zobs
from ..cosmology.geometry_utils import compute_volume_from_sky_area
from ..hmf import mc_hosts, hmf_model
from ..mah.utils import apply_mah_rescaling
from ..defaults import DELTA_C

N_HMF_GRID = 500
DEFAULT_LOGMP_CUTOFF = 10.0
DEFAULT_LOGMP_HIMASS_CUTOFF = 14.5

DEFAULT_DIFFMAHNET_CEN_MODEL = "cenflow_v2_0_64bit.eqx"

_AXES = (0, None, None, 0, None, None)
mc_logmp_vmap = jjit(vmap(mc_hosts._mc_host_halos_singlez_kern, in_axes=_AXES))

__all__ = ("mc_lc_hmf", "mc_lc_halos", "weighted_lc_halos")

_CENPOP_FIELDS = (
    "z_obs",
    "t_obs",
    "logmp_obs",
    "mah_params",
    "logmp0",
    "logt0",
    "nhalos",
)
CenPop = namedtuple("CenPop", _CENPOP_FIELDS)


def mc_lc_hmf(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    cosmo_params=DEFAULT_COSMOLOGY,
    lgmp_max=mc_hosts.LGMH_MAX,
    n_hmf_grid=N_HMF_GRID,
    delta_c=DELTA_C,
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

    cosmo_params: jax-cosmo parameters object
        cosmological parameters

    lgmp_max: float
        base-10 log of maximum halo mass, in Msun

    n_hmf_grid: int
        number of redshift grid points for HMF computations

    delta_c: float
        overdensity threshold

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
    mean_nhalos = compute_mean_halo(
        cosmo_params, lgmp_min, lgmp_max, z_grid, volume_com_mpc, delta_c
    )

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

    # draw halo masses from the HMF CDF
    logmp_cenpop = mc_logmp_vmap(
        uran_m, cosmo_params, lgmp_min, z_cenpop, lgmp_max, delta_c
    )

    return z_cenpop, logmp_cenpop


@jjit
def _compute_mean_halo_kern(
    cosmo_params,
    lgmp_min,
    lgmp_max,
    z,
    volume_com_mpc,
    delta_c,
):
    # at each grid point, compute <Nhalos> for the shell volume
    mean_nhalos = hmf_model._compute_nhalos_tot(
        cosmo_params, lgmp_min, z, volume_com_mpc, delta_c=delta_c
    )
    mean_nhalos_lgmax = hmf_model._compute_nhalos_tot(
        cosmo_params, lgmp_max, z, volume_com_mpc, delta_c=delta_c
    )
    mean_nhalos = mean_nhalos - mean_nhalos_lgmax

    return mean_nhalos


compute_mean_halo = jjit(
    vmap(_compute_mean_halo_kern, in_axes=(None, None, None, 0, 0, None))
)


def mc_lc_halos(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    cosmo_params=DEFAULT_COSMOLOGY,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
    lgmp_max=mc_hosts.LGMH_MAX,
    n_hmf_grid=N_HMF_GRID,
    centrals_model_key=DEFAULT_DIFFMAHNET_CEN_MODEL,
    delta_c=DELTA_C,
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

    cosmo_params: jax-cosmo parameters object
        cosmological parameters

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

    delta_c: float
        overdensity threshold

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
        lgmp_max=lgmp_max,
        n_hmf_grid=n_hmf_grid,
        delta_c=delta_c,
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
    fields = CenPop._fields[:-1]
    values = (z_obs, t_obs, logmp_obs, mah_params, logmp0, logt0)
    cenpop = namedtuple("cenpop", fields)(*values)

    return cenpop


def weighted_lc_halos(
    ran_key,
    n_halos,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    *,
    cosmo_params=DEFAULT_COSMOLOGY,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
    centrals_model_key=DEFAULT_DIFFMAHNET_CEN_MODEL,
    delta_c=DELTA_C,
):
    """
    Generate a mass-function-weighted lightcone of halos and their
    mass assembly histories.

    Parameters
    ----------
    ran_key: jran.key
        random key

    n_halos : int
        Number of host halos in the weighted lightcone

    z_min, z_max : float
        min/max redshift

    lgmp_min,lgmp_max : float
        log10 of min/max halo mass in units of Msun

    sky_area_degsq: float
        sky area in deg^2

    cosmo_params: jax-cosmo parameters object
        cosmological parameters

    logmp_cutoff: float, optional kwarg
        base-10 log of minimum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun;
        for logmp < logmp_cutoff, P(θ_MAH | logmp) = P(θ_MAH | logmp_cutoff)

    logmp_cutoff_himass: float, optional kwarg
        base-10 log of maximum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun

    delta_c: float
        overdensity threshold

    Returns
    -------
    halopop: namedtuple
        Population of n_halos halos

        halopop fields:
            z_obs: ndarray of shape (n_halos, )
                redshift values

            t_obs: ndarray of shape (n_halos, )
                cosmic time at observation, in Gyr

            logmp_obs: ndarray of shape (n_halos, )
                base-10 log of halo mass at observation, in Msun

            mah_params: namedtuple of ndarrays of shape (n_halos, )
                mah parameters

            logmp0: ndarray of shape (n_halos, )
                base-10 log of halo mass at z=0, in Msun

            logt0: float
                Base-10 log of z=0 age of the Universe for the input cosmology

            nhalos: ndarray of shape (n_halos, )
                weight of the (sub)halo

    """
    lgm_key, redshift_key, halo_key = jran.split(ran_key, 3)
    logmp_obs = jran.uniform(
        lgm_key, minval=lgmp_min, maxval=lgmp_max, shape=(n_halos,)
    )
    z_obs = jran.uniform(redshift_key, minval=z_min, maxval=z_max, shape=(n_halos,))

    cenpop = _weighted_lc_halos_from_grid(
        halo_key,
        z_obs,
        logmp_obs,
        sky_area_degsq,
        cosmo_params,
        logmp_cutoff,
        logmp_cutoff_himass,
        centrals_model_key,
        delta_c,
    )
    return cenpop


@partial(jjit, static_argnames=["centrals_model_key"])
def _weighted_lc_halos_from_grid(
    ran_key,
    z_obs,
    logmp_obs,
    sky_area_degsq,
    cosmo_params=DEFAULT_COSMOLOGY,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
    centrals_model_key=DEFAULT_DIFFMAHNET_CEN_MODEL,
    delta_c=DELTA_C,
):
    # get halo weights
    nhalo_weights = hmf_model.halo_lightcone_weights(
        logmp_obs,
        z_obs,
        sky_area_degsq,
        cosmo_params=cosmo_params,
        delta_c=delta_c,
    )

    t_obs, t_0 = get_tobs_from_zobs(z_obs, cosmo_params=cosmo_params)
    logt0 = jnp.log10(t_0)

    # get rescaled mah parameters and mah values at t_obs
    logmp_obs_clipped = jnp.clip(logmp_obs, logmp_cutoff, logmp_cutoff_himass)
    mah_params, logmp_obs = apply_mah_rescaling(
        ran_key,
        logmp_obs,
        logmp_obs_clipped,
        t_obs,
        logt0,
        centrals_model_key,
    )

    # compute mah values today
    logmp0 = _log_mah_kern(mah_params, 10**logt0, logt0)

    # create output namedtuple
    values = (z_obs, t_obs, logmp_obs, mah_params, logmp0, logt0, nhalo_weights)
    cenpop = CenPop(*values)

    return cenpop
