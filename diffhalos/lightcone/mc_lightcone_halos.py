# flake8: noqa: E402
"""Functions to generate host halo lightcones"""

from jax import config

config.update("jax_enable_x64", True)

from collections import namedtuple
from functools import partial

from diffmah.diffmah_kernels import _log_mah_kern
from dsps.cosmology import DEFAULT_COSMOLOGY, flat_wcdm
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..hmf import hmf_model, mc_hosts
from ..hmf.hmf_model import halo_lightcone_weights
from ..mah.diffmahnet_utils import mc_mah_cenpop as mc_mah_cenpop_diffmahnet

N_HMF_GRID = 2_000
DEFAULT_LOGMP_CUTOFF = 10.0
DEFAULT_LOGMP_HIMASS_CUTOFF = 14.5

DEFAULT_DIFFMAHNET_CEN_MODEL = "cenflow_v2_0_64bit.eqx"

_AXES = (0, None, None, 0, None)
mc_logmp_vmap = jjit(vmap(mc_hosts._mc_host_halos_singlez_kern, in_axes=_AXES))

__all__ = (
    "mc_lightcone_host_halo",
    "weighted_lightcone_host_halo",
)


def mc_lc_hmf(ran_key, lgmp_min, lgmp_max, z_min, z_max, sky_area_degsq):
    return z_halopop, logmp_halopop


def mc_lc_halos(ran_key, lgmp_min, lgmp_max, z_min, z_max, sky_area_degsq):
    z_halopop, logmp_halopop = mc_lc_hmf(
        ran_key, lgmp_min, lgmp_max, z_min, z_max, sky_area_degsq
    )
    cenpop = mc_lightcone_host_halo(*args)
    return cenpop


@partial(jjit, static_argnames=("nhalos_tot",))
def mc_lightcone_host_halo_mass_function(
    ran_key,
    lgmp_min,
    z_grid,
    sky_area_degsq,
    nhalos_tot,
    cosmo_params=DEFAULT_COSMOLOGY,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    lgmp_max=mc_hosts.LGMH_MAX,
):
    """
    Generate a Monte Carlo realization of a lightcone of
    host halo masses and redshifts, on an input grid
    in redshift, between a minimum and a maximum halo mass
    sampling from the halo mass function

    Parameters
    ----------
    ran_key: jran.key
        random key

    lgmp_min: float
        minimum halo mass, in Msun

    z_grid: ndarray of shape (n_z, )
        redshift points

    sky_area_degsq: float
        sky area, in deg^2

    nhalos_tot: int
        total number of halos to generate in the lightcone

    cosmo_params: namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    hmf_params: namedtuple
        halo mass function parameters

    lgmp_min: float
        base-10 log of minimum halo mass, in Msun

    lgmp_max: float
        base-10 log of maximum halo mass, in Msun

    Returns
    -------
    z_halopop: ndarray of shape (n_halos, )
        redshifts distributed randomly within the lightcone volume

    logmp_halopop: ndarray of shape (n_halos, )
        halo masses derived by Monte Carlo sampling the halo mass function
        at the appropriate redshift for each point
    """

    m_key, z_key = jran.split(ran_key, 2)

    # get the mean number of halos on redshift grid
    mean_nhalos_grid = hmf_model.get_mean_nhalos_from_sky_area(
        z_grid,
        sky_area_degsq,
        cosmo_params,
        hmf_params,
        lgmp_min,
        lgmp_max,
    )

    # compute the CDF of the volume
    weights_grid = mean_nhalos_grid / mean_nhalos_grid.sum()
    cdf_grid = jnp.cumsum(weights_grid)

    # assign redshift via inverse transformation sampling of the halo counts CDF
    uran_z = jran.uniform(z_key, minval=0, maxval=1, shape=(nhalos_tot,))
    z_halopop = jnp.interp(uran_z, cdf_grid, z_grid)

    # randoms used in inverse transformation sampling halo mass
    uran_m = jran.uniform(m_key, minval=0, maxval=1, shape=(nhalos_tot,))

    # draw a halo mass from the HMF at the particular redshift of each halo
    logmp_halopop = mc_logmp_vmap(uran_m, hmf_params, lgmp_min, z_halopop, lgmp_max)

    return z_halopop, logmp_halopop


@partial(jjit, static_argnames=["centrals_model_key", "nhalos_tot"])
def mc_lightcone_host_halo(
    ran_key,
    lgmp_min,
    z_grid,
    sky_area_degsq,
    nhalos_tot,
    cosmo_params=DEFAULT_COSMOLOGY,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
    lgmp_max=mc_hosts.LGMH_MAX,
    centrals_model_key=DEFAULT_DIFFMAHNET_CEN_MODEL,
):
    """
    Generate a halo lightcone, including MAHs, on an input
    grid in redshift, between a minimum and a maximum halo mass,
    via sampling from the halo mass function

    Parameters
    ----------
    ran_key: jran.key
        random key

    lgmp_min: float
        minimum halo mass, in Msun

    z_grid: ndarray of shape (n_z, )
        redshift values

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
        DiffmahPop is used to generate MAHs, in Msun;
        for logmp < logmp_cutoff, P(θ_MAH | logmp) = P(θ_MAH | logmp_cutoff)

    logmp_cutoff_himass: float
        base-10 log of maximum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun

    lgmp_max: float
        base-10 log of maximum host halo mass, in Msun

    centrals_model_key: str
        diffmahnet model to use for centrals

    Returns
    -------
    cenpop: namedtuple with fields:
        z_obs: ndarray of shape (n_halos, )
            lightcone redshift

        logmp_obs: ndarray of shape (n_halos, )
            halo mass at the lightcone redshift, in Msun

        mah_params: namedtuple of ndarray's with shape (n_halos, n_mah_params)
            diffmah parameters for each host halo in the lightcone

        logmp0: narray of shape (n_halos, )
            base-10 log of halo mass at z=0, in Msun
    """

    # generate mc realization of the halo mass function
    lc_hmf_key, mah_key = jran.split(ran_key, 2)
    z_obs, logmp_obs_mf = mc_lightcone_host_halo_mass_function(
        lc_hmf_key,
        lgmp_min,
        z_grid,
        sky_area_degsq,
        nhalos_tot,
        cosmo_params=cosmo_params,
        hmf_params=hmf_params,
        lgmp_max=lgmp_max,
    )
    t_obs = flat_wcdm.age_at_z(z_obs, *cosmo_params)
    t_0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t_0)
    logmp_obs_mf_clipped = jnp.clip(logmp_obs_mf, logmp_cutoff, logmp_cutoff_himass)

    # get the MAH parameters for the halos
    num_halos = t_obs.size
    tarr = (jnp.ones(num_halos) * lgt0).reshape(num_halos, 1)
    logmp_obs, mah_params = mc_mah_cenpop_diffmahnet(
        logmp_obs_mf_clipped,
        t_obs,
        mah_key,
        tarr,
        centrals_model_key=centrals_model_key,
        logt0=lgt0,
    )
    logmp_obs = jnp.concatenate(logmp_obs)

    # compute MAH values today
    logmp0 = _log_mah_kern(mah_params, 10**lgt0, lgt0)

    # create output dictionary
    fields = ("z_obs", "t_obs", "logmp_obs", "mah_params", "logmp0")
    values = (z_obs, t_obs, logmp_obs, mah_params, logmp0)
    cenpop_out = namedtuple("halopop", fields)(*values)

    return cenpop_out


@partial(jjit, static_argnames=["centrals_model_key"])
def weighted_lightcone_host_halo(
    ran_key,
    z_obs,
    logmp_obs_mf,
    sky_area_degsq,
    cosmo_params=DEFAULT_COSMOLOGY,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
    centrals_model_key=DEFAULT_DIFFMAHNET_CEN_MODEL,
):
    """
    Generates a weighted lightcone population of halos with MAHs,
    on a grid generated based on a sequence of any kind

    Parameters
    ----------
    ran_key: jran.key
        random key

    z_obs: ndarray of shape (n_halo, )
        observed redshifts of galaxies

    logmp_obs_mf: ndarray of shape (n_halo, )
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
    cenpop_out: namedtuple with fields:
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
        logmp_obs_mf,
        z_obs,
        sky_area_degsq,
        hmf_params=hmf_params,
        cosmo_params=cosmo_params,
    )
    t_obs = flat_wcdm.age_at_z(z_obs, *cosmo_params)
    t_0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t_0)

    logmp_obs_clipped = jnp.clip(logmp_obs_mf, logmp_cutoff, logmp_cutoff_himass)

    tarr = jnp.array((10**lgt0,))

    # get the MAH parameters for the halos
    num_halos = t_obs.size
    ran_key, mah_key = jran.split(ran_key, 2)
    tarr = (jnp.ones(num_halos) * lgt0).reshape(num_halos, 1)
    logmp_obs, mah_params = mc_mah_cenpop_diffmahnet(
        logmp_obs_clipped,
        t_obs,
        mah_key,
        tarr,
        centrals_model_key=centrals_model_key,
        logt0=lgt0,
    )
    logmp_obs = jnp.concatenate(logmp_obs)

    # compute MAH values today
    logmp0 = _log_mah_kern(mah_params, 10**lgt0, lgt0)

    # create output namedtuple
    fields = ("z_obs", "t_obs", "logmp_obs", "mah_params", "logmp0", "nhalos")
    values = (z_obs, t_obs, logmp_obs, mah_params, logmp0, nhalo_weights)
    cenpop_out = namedtuple("halopop", fields)(*values)

    return cenpop_out
