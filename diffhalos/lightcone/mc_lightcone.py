# flake8: noqa: E402
"""Generation of halo plus subhalo lightcones.
This module combines the halo and subhalo lightcone generators
and combines the outputs in a user-friendly and useful way."""

from jax import config

config.update("jax_enable_x64", True)

from jax import random as jran

from . import mc_lightcone_halos as mclch
from . import mc_lightcone_subhalos as mclcsh

from ..hmf import mc_hosts
from ..cosmology import DEFAULT_COSMOLOGY
from ..ccshmf import DEFAULT_CCSHMF_PARAMS

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
            z_obs: ndarray of shape (n_halos, )
                host halo redshifts

            logmp_obs: ndarray of shape (n_halos, )
                host halo mass at the lightcone redshifts, in Msun

            logmp0: narray of shape (n_halos, )
                base-10 log of host halo mass at z=0, in Msun

            logt0: float
                base-10 log of cosmic time at z=0, in Gyr

            nsub_per_host: ndarray of shape (n_host, )
                number of generated subhalos per host halo

            logmu_obs: ndarray of shape (n_subs, )
                base-10 log of mu=Msub/Mhost for each subhalo in the lightcone

            mah_params: namedtuple of ndarray's with shape (n_halos, n_mah_params)
                MAH parameters for each host halo in the lightcone

            mah_params_subs: namedtuple of ndarray's with shape (n_subs, n_mah_params)
                diffmah parameters for each subhalo in the lightcone
    """
    # two random keys, one for the host and one for the subhalo population
    host_key, subhalo_key = jran.split(ran_key)

    # generate a host halo lightcone
    cenpop = mclcsh.mc_lc_halos(
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

    return


def weighted_lc(
    ran_key,
    z_obs,
    logmp_obs,
    lgmsub_min,
    sky_area_degsq,
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
):
    """
    Generates a weighted lightcone population of halos+subhalos with MAHs,
    on an input grid of redshift and mass

    Parameters
    ----------
    ran_key: jran.key
        random key

    z_obs: ndarray of shape (n_halo, )
        observed redshifts of galaxies

    logmp_obs: ndarray of shape (n_halo, )
        base-10 log of observed halo masses, in Msun

    lgmsub_min: float
        base-10 log of the minimum mass, in Msun

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

    n_mu_per_host: int
        number of mu=Msub/Mhost values to use per host halo;
        note that for the weighted version of the lightcone,
        each host gets assigned the same number of subhalos

    cshmf_params: namedtuple
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
            z_obs: ndarray of shape (n_halo, )
                redshift values

            t_obs: ndarray of shape (n_halo, )
                cosmic time at observation, in Gyr

            logmp_obs: ndarray of shape (n_halo, )
                base-10 log of halo mass at observation, in Msun

            logmp0: ndarray of shape (n_halo, )
                base-10 log of halo mass at z=0, in Msun

            nhalos: ndarray of shape (n_halo, )
                weighted number of halos at each grid point

            nsubhalos: ndarray of shape (n_nub, )
                subhalo weighted counts

            logmu_obs: ndarray of shape (n_subs, )
                base-10 log of mu=Msub/Mhost for each subhalo in the lightcone

            nsub_per_host: int
                number of subhalo points generated per host halo

            mah_params_subs: namedtuple of ndarray's with shape (n_subs, n_mah_params)
                diffmah parameters for each subhalo in the lightcone

            mah_params: namedtuple of ndarrays of shape (n_halo, )
                mah parameters
    """
    # two random keys, one for the host and one for the subhalo population
    host_key, subhalo_key = jran.split(ran_key)

    # generate a weighted host halo lightcone
    cenpop = mclch.weighted_lc_halos(
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

    return
