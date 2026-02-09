# flake8: noqa: E402
"""Generation of halo plus subhalo lightcones.
This module combines the halo and subhalo lightcone generators
and combines the outputs in a user-friendly and useful way."""

from jax import config

config.update("jax_enable_x64", True)

from collections import namedtuple

import numpy as np
from diffmah.diffmah_kernels import _log_mah_kern
from jax import numpy as jnp
from jax import random as jran

from ..ccshmf import DEFAULT_CCSHMF_PARAMS
from ..cosmology import DEFAULT_COSMOLOGY
from ..hmf import mc_hosts
from . import mc_lightcone_halos as mclch
from . import mc_lightcone_subhalos as mclcsh

__all__ = ("mc_lc_mf", "mc_lc", "weighted_lc")


def mc_lc_mf(
    ran_key,
    lgmp_min,
    lgmsub_min,
    z_min,
    z_max,
    sky_area_degsq,
    cosmo_params=DEFAULT_COSMOLOGY,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    lgmp_max=mc_hosts.LGMH_MAX,
    n_hmf_grid=mclch.N_HMF_GRID,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
):
    """
    Generate a Monte Carlo realization of a lightcone of
    host halos and subhalos

    Parameters
    ----------
    ran_key: jran.key
        random key

    lgmp_min: float
        minimum host halo mass, in Msun

    lgmsub_min: float
        minimum subhalo mass, in Msun

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

    ccshmf_params: namedtuple
        CCSHMF parameters

    Returns
    -------
    z_halopop: ndarray of shape (n_halos, )
        redshifts distributed randomly within the lightcone volume

    logmp_halopop: ndarray of shape (n_halos, )
        halo masses derived by Monte Carlo sampling the halo mass function
        at the appropriate redshift for each point
    """
    # two random keys, one for the host and one for the subhalo population
    host_key, subhalo_key = jran.split(ran_key)

    # generate host halo MC lightcone
    z_halopop, logmp_cenpop = mclch.mc_lc_hmf(
        host_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
        cosmo_params=cosmo_params,
        hmf_params=hmf_params,
        lgmp_max=lgmp_max,
        n_hmf_grid=n_hmf_grid,
    )

    # generate subhalo MC lightcone using the host halo realization
    mc_lg_mu, subhalo_counts_per_halo = mclcsh.mc_lc_shmf(
        subhalo_key,
        logmp_cenpop,
        lgmsub_min,
        ccshmf_params=ccshmf_params,
    )

    return z_halopop, logmp_cenpop, mc_lg_mu, subhalo_counts_per_halo


def mc_lc(
    ran_key,
    lgmp_min,
    lgmsub_min,
    z_min,
    z_max,
    sky_area_degsq,
    cosmo_params=DEFAULT_COSMOLOGY,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    logmp_cutoff=mclch.DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=mclch.DEFAULT_LOGMP_HIMASS_CUTOFF,
    lgmp_max=mc_hosts.LGMH_MAX,
    n_hmf_grid=mclch.N_HMF_GRID,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
    logmsub_cutoff=mclcsh.DEFAULT_LOGMSUB_CUTOFF,
    logmsub_cutoff_himass=mclcsh.DEFAULT_LOGMSUB_HIMASS_CUTOFF,
    centrals_model_key=mclch.DEFAULT_DIFFMAHNET_CEN_MODEL,
    subhalo_model_key=mclcsh.DEFAULT_DIFFMAHNET_SAT_MODEL,
):
    """
    Generate a halo+subhalo lightcone, including MAHs,
    between a minimum and a maximum value of redshift and halo mass

    Parameters
    ----------
    ran_key: jran.key
        random key

    lgmp_min: float
        minimum halo mass, in Msun

    lgmsub_min: float
        base-10 log of the minimum mass, in Msun

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

    ccshmf_params: namedtuple
        CCSHMF parameters

    logmsub_cutoff: float
        base-10 log of minimum subhalo mass for which
        diffmahnet is used to generate MAHs, in Msun;
        for logmsub < logmsub_cutoff, P(θ_MAH | logmsub) = P(θ_MAH | logmsub_cutoff)

    logmsub_cutoff_himass: float
        base-10 log of maximum subhalo mass for which
        diffmahnet is used to generate MAHs, in Msun

    centrals_model_key: str
        diffmahnet model to use for centrals

    subhalo_model_key: str
        diffmahnet model to use for satellites

    Returns
    -------
    halopop: namedtuple
        halo population with fields:
            z_obs: ndarray of shape (n_host, )
                lightcone redshift

            logmp_obs: ndarray of shape (n_host, )
                halo mass at the lightcone redshift, in Msun

            mah_params: namedtuple of ndarray's with shape (n_host+n_sub, n_mah_params)
                diffmah parameters for each host halo in the lightcone

            logmp0: narray of shape (n_host, )
                base-10 log of halo mass at z=0, in Msun

            logt0: float
                base-10 log of cosmic time at today, in Gyr

            nsub_per_host: ndarray of shape (n_host, )
                number of subhalos generated per host halo

            logmu_obs: ndarray of shape (n_sub, )
                base-10 log of mu=Msub/Mhost for each generated subhalo

            halo_indx: ndarray of shape (n_host+n_sub, )
                halo index; for host halos, the index is arange(n_host),
                while for the subhalos, indeces correspond to the host
                halo that hosts each generated subhalo
    """
    # two random keys, one for the host and one for the subhalo population
    host_key, subhalo_key = jran.split(ran_key)

    # generate a host halo lightcone
    cenpop = mclch.mc_lc_halos(
        host_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
        cosmo_params=cosmo_params,
        hmf_params=hmf_params,
        logmp_cutoff=logmp_cutoff,
        logmp_cutoff_himass=logmp_cutoff_himass,
        lgmp_max=lgmp_max,
        n_hmf_grid=n_hmf_grid,
        centrals_model_key=centrals_model_key,
    )
    # fields = ("z_obs", "t_obs", "logmp_obs", "mah_params", "logmp0", "logt0")

    # generate a subhalo lightcone
    subpop = mclcsh.mc_lc_subhalos(
        subhalo_key,
        cenpop,
        lgmsub_min,
        ccshmf_params=ccshmf_params,
        logmsub_cutoff=logmsub_cutoff,
        logmsub_cutoff_himass=logmsub_cutoff_himass,
        subhalo_model_key=subhalo_model_key,
    )

    # create the index array: [...host_indx..., ...subhalo_indx...]
    n_host = cenpop.logmp_obs.size
    host_indx = jnp.arange(n_host).astype(int)
    n_sub = int(subpop.nsub_per_host.sum())
    subhalo_indx = jnp.repeat(host_indx, subpop.nsub_per_host)
    halo_indx = jnp.concatenate((host_indx, subhalo_indx)).astype(int)

    # combine halo and subhalo mah_params
    mah_params_names = cenpop.mah_params._fields
    mah_params_tot = np.zeros((len(mah_params_names), n_host + n_sub))
    for i, _param in enumerate(mah_params_names):
        mah_params_tot[i, :] = np.concatenate(
            (
                cenpop.mah_params._asdict()[_param],
                subpop.mah_params._asdict()[_param],
            )
        )
    mah_params_ntup = namedtuple("mah_params", cenpop.mah_params._fields)(
        *mah_params_tot
    )
    cenpop = cenpop._replace(mah_params=mah_params_ntup)

    # create the output namedtuple containing host and subhalo information;
    # this will contain all host halo information, updated to include
    # the subhalo information and some fields are updated to new shapes
    halopop = namedtuple(
        "mc_lc", [*cenpop._fields, "nsub_per_host", "logmu_obs", "halo_indx"]
    )(*cenpop, subpop.nsub_per_host, subpop.logmu_obs, halo_indx)

    return halopop


def weighted_lc(
    ran_key,
    n_host_halos,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    *,
    cosmo_params=DEFAULT_COSMOLOGY,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    logmp_cutoff=mclch.DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=mclch.DEFAULT_LOGMP_HIMASS_CUTOFF,
    n_mu_per_host=mclcsh.N_LGMU_PER_HOST,
    ccshmf_params=mclcsh.DEFAULT_CCSHMF_PARAMS,
    logmsub_cutoff=mclcsh.DEFAULT_LOGMSUB_CUTOFF,
    logmsub_cutoff_himass=mclcsh.DEFAULT_LOGMSUB_HIMASS_CUTOFF,
    centrals_model_key=mclch.DEFAULT_DIFFMAHNET_CEN_MODEL,
    subhalo_model_key=mclcsh.DEFAULT_DIFFMAHNET_SAT_MODEL,
    lgmsub_min=None,
):
    """
    Generate a mass-function-weighted lightcone of halos+subhalos and their
    mass assembly histories.

    Parameters
    ----------
    ran_key: jran.key
        random key

    n_host_halos : int
        Number of host halos in the weighted lightcone

    z_min, z_max : float
        min/max redshift

    lgmp_min,lgmp_max : float
        log10 of min/max halo mass in units of Msun

    sky_area_degsq: float
        sky area in deg^2

    cosmo_params: namedtuple, optional kwarg
        cosmological parameters

    hmf_params: namedtuple, optional kwarg
        halo mass function parameters

    logmp_cutoff: float, optional kwarg
        base-10 log of minimum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun;
        for logmp < logmp_cutoff, P(θ_MAH | logmp) = P(θ_MAH | logmp_cutoff)

    logmp_cutoff_himass: float, optional kwarg
        base-10 log of maximum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun

    n_mu_per_host: int, optional kwarg
        number of mu=Msub/Mhost values to use per host halo;
        note that for the weighted version of the lightcone,
        each host gets assigned the same number of subhalos

    cshmf_params: namedtuple, optional kwarg
        CCSHMF parameters

    logmsub_cutoff: float, optional kwarg
        base-10 log of minimum subhalo mass for which
        diffmahnet is used to generate MAHs, in Msun;
        for logmsub < logmsub_cutoff, P(θ_MAH | logmsub) = P(θ_MAH | logmsub_cutoff)

    logmsub_cutoff_himass: float, optional kwarg
        base-10 log of maximum subhalo mass for which
        diffmahnet is used to generate MAHs, in Msun

    centrals_model_key: str, optional kwarg
        diffmahnet model to use for centrals

    subhalo_model_key: str, optional kwarg
        diffmahnet model to use for satellites

    lgmsub_min: float, optional kwarg
        base-10 log of the minimum subhalo mass, in Msun
        If none, will be set to lgmp_min-ε

    Returns
    -------
    halopop: namedtuple
        Population of n_halos_tot halos and subhalos
            n_halos_tot = n_sub + n_host_halos
            n_sub = nsub_per_host * n_host_halos

        halopop fields:
            z_obs: ndarray of shape (n_halos_tot, )
                redshift values

            t_obs: ndarray of shape (n_halos_tot, )
                cosmic time at observation, in Gyr

            logmp_obs: ndarray of shape (n_halos_tot, )
                base-10 log of halo mass at observation, in Msun

            mah_params: namedtuple of ndarrays of shape (n_halos_tot, )
                mah parameters

            logmp0: ndarray of shape (n_halos_tot, )
                base-10 log of halo mass at z=0, in Msun

            logt0: float
                Base-10 log of z=0 age of the Universe for the input cosmology

            nhalos: ndarray of shape (n_halos_tot, )
                weight of the (sub)halo

            nhalos_host: ndarray of shape (n_halos_tot, )
                weight of the host halo
                Equal to nhalos for central halos

            nsub_per_host: int
                number of subhalos per host halo
                    n_sub = nsub_per_host * n_host_halos
                    n_halos_tot = n_sub + n_host_halos

            logmu_obs: ndarray of shape (n_halos_tot, )
                base-10 log of mu=Msub/Mhost

            halo_indx: ndarray of shape (n_halos_tot, )
                index of the associated host halo
                for central halos: halo_indx = range(n_halos_tot)

    """
    lgm_key, redshift_key, halo_key = jran.split(ran_key, 3)
    logmp_obs = jran.uniform(
        lgm_key, minval=lgmp_min, maxval=lgmp_max, shape=(n_host_halos,)
    )
    z_obs = jran.uniform(
        redshift_key, minval=z_min, maxval=z_max, shape=(n_host_halos,)
    )

    if lgmsub_min is None:
        lgmsub_min = lgmp_min - 0.01

    halopop = _weighted_lc_from_grid(
        halo_key,
        z_obs,
        logmp_obs,
        lgmsub_min,
        sky_area_degsq,
        cosmo_params,
        hmf_params,
        logmp_cutoff,
        logmp_cutoff_himass,
        n_mu_per_host,
        ccshmf_params,
        logmsub_cutoff,
        logmsub_cutoff_himass,
        centrals_model_key,
        subhalo_model_key,
    )
    return halopop


def _weighted_lc_from_grid(
    ran_key,
    z_obs,
    logmp_obs,
    lgmsub_min,
    sky_area_degsq,
    cosmo_params,
    hmf_params,
    logmp_cutoff,
    logmp_cutoff_himass,
    n_mu_per_host,
    ccshmf_params,
    logmsub_cutoff,
    logmsub_cutoff_himass,
    centrals_model_key,
    subhalo_model_key,
):
    # two random keys, one for the host and one for the subhalo population
    host_key, subhalo_key = jran.split(ran_key)

    # generate a weighted host halo lightcone
    cenpop = mclch._weighted_lc_halos_from_grid(
        host_key,
        z_obs,
        logmp_obs,
        sky_area_degsq,
        cosmo_params=cosmo_params,
        hmf_params=hmf_params,
        logmp_cutoff=logmp_cutoff,
        logmp_cutoff_himass=logmp_cutoff_himass,
        centrals_model_key=centrals_model_key,
    )

    # generate a weighted subhalo lightcone
    subpop = mclcsh.weighted_lc_subhalos(
        subhalo_key,
        cenpop,
        lgmsub_min,
        n_mu_per_host=n_mu_per_host,
        ccshmf_params=ccshmf_params,
        logmsub_cutoff=logmsub_cutoff,
        logmsub_cutoff_himass=logmsub_cutoff_himass,
        subhalo_model_key=subhalo_model_key,
    )

    # create the index array
    n_host = cenpop.logmp_obs.size
    n_sub = subpop.logmp_obs.size
    host_indx = jnp.arange(n_host).astype(int)
    subhalo_indx = jnp.repeat(host_indx, subpop.nsub_per_host)
    halo_indx = jnp.concatenate((host_indx, subhalo_indx)).astype(int)

    z_obs_subs = jnp.repeat(cenpop.z_obs, subpop.nsub_per_host)
    z_obs_all = jnp.concatenate((cenpop.z_obs, z_obs_subs))
    cenpop = cenpop._replace(z_obs=z_obs_all)

    t_obs_subs = jnp.repeat(cenpop.t_obs, subpop.nsub_per_host)
    t_obs_all = jnp.concatenate((cenpop.t_obs, t_obs_subs))
    cenpop = cenpop._replace(t_obs=t_obs_all)

    nhalos_host_subs = jnp.repeat(cenpop.nhalos, subpop.nsub_per_host)
    nhalos_host_all = jnp.concatenate((cenpop.nhalos, nhalos_host_subs))

    logmp_obs_all = jnp.concatenate((cenpop.logmp_obs, subpop.logmp_obs))
    cenpop = cenpop._replace(logmp_obs=logmp_obs_all)

    # compute mah values at z=0 for subs
    logmp0_subs = _log_mah_kern(subpop.mah_params, 10**cenpop.logt0, cenpop.logt0)
    logmp0_all = jnp.concatenate((cenpop.logmp0, logmp0_subs))
    cenpop = cenpop._replace(logmp0=logmp0_all)

    # combine halo and subhalo mah_params
    mah_params_names = cenpop.mah_params._fields
    mah_params_tot = np.zeros((len(mah_params_names), n_host + n_sub))
    for i, _param in enumerate(mah_params_names):
        mah_params_tot[i, :] = np.concatenate(
            (
                cenpop.mah_params._asdict()[_param],
                subpop.mah_params._asdict()[_param],
            )
        )
    mah_params_ntup = namedtuple("mah_params", cenpop.mah_params._fields)(
        *mah_params_tot
    )
    cenpop = cenpop._replace(mah_params=mah_params_ntup)

    # combine halo and subhalo weights
    cenpop = cenpop._replace(nhalos=np.concatenate((cenpop.nhalos, subpop.nsubhalos)))

    logmu_obs_host = jnp.zeros(n_host)
    logmu_obs_all = jnp.concatenate((logmu_obs_host, subpop.logmu_obs))

    # create the output namedtuple containing host and subhalo information;
    # this will contain all host halo information, updated to include
    # the subhalo information and some fields are updated to new shapes
    halopop = namedtuple(
        "weighted_lc",
        [*cenpop._fields, "nhalos_host", "nsub_per_host", "logmu_obs", "halo_indx"],
    )(*cenpop, nhalos_host_all, subpop.nsub_per_host, logmu_obs_all, halo_indx)

    return halopop
