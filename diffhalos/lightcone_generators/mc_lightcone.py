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

# TODO: I think we could also define halo_weight == gal_weight = cen_weight * sat_weight here, instead of leaving the user to multiply afterwards.
_HALOPOP_FIELDS = (
    "z_obs",  # defined for centrals, repeated to be assigned to all objects
    "t_obs",  # defined for centrals, repeated to be assigned to all objects
    "logmp_obs",  # join the values of cens and subs
    "mah_params",  # join the values of cens and subs
    "logmp0",  # join the values of cens and subs
    "logt0",  # defined for centrals, repeated to be assigned to all objects
    "cen_weight",  # defined for centrals, repeated to be assigned to all objects
    "central",  # host halo idenfier
    "sat_weight",  # defined for subs, repeated to be assigned to all objects
    "nsub_per_host",
    "logmu_obs",  # defined for subs, repeated to be assigned to all objects ?
    "halo_indx",
)

HaloPop = namedtuple("HaloPop", _HALOPOP_FIELDS)


def _combine_cenpop_subpop(cenpop, subpop):
    """
    Auxiliary function to reshape and combine cens and subs quantities, and build an unified halopop namedtuple.

    Returns halopop
    This is almost the same as simply joining CenPop and SubPop, but some variables defined for each namedtuple separately are combined here into a single one, e.g., MAH params, logmp_obs. So they do not appear twice, as they would if we were simply joining CenPop and SubPop.

    Paste here full description of halopop
    """

    # Create the halo index array
    n_host = cenpop.logmp_obs.size
    n_sub = subpop.logmp_obs.size
    host_indx = jnp.arange(n_host).astype(int)
    subhalo_indx = jnp.repeat(host_indx, subpop.nsub_per_host)
    halo_indx = jnp.concatenate((host_indx, subhalo_indx)).astype(int)

    # Create the central identifier array
    is_central = jnp.concatenate((jnp.ones(n_host), jnp.zeros(n_sub))).astype(int)

    # -- Combine properties --

    # Reshape z_obs to assign values for all halos
    z_obs_all = jnp.concatenate(
        (cenpop.z_obs, jnp.repeat(cenpop.z_obs, subpop.nsub_per_host))
    )

    # Reshape t_obs to assign values for all halos
    t_obs_all = jnp.concatenate(
        (cenpop.t_obs, jnp.repeat(cenpop.t_obs, subpop.nsub_per_host))
    )

    # Combine logmp_obs from cens and subs
    logmp_obs_all = jnp.concatenate((cenpop.logmp_obs, subpop.logmp_obs))

    # Combine logmp0 from cens and subs
    # Compute mah values at z=0 for subs
    logmp0_subs = _log_mah_kern(subpop.mah_params, 10**cenpop.logt0, cenpop.logt0)
    logmp0_all = jnp.concatenate((cenpop.logmp0, logmp0_subs))
    # TODO: logmp0_all = jnp.concatenate((cenpop.logmp0, subpop.logmp0_subs))
    # This implies adding the logmp0 field to SubPop and running the mah kernel in the subs function instead of doing it here.

    # Reshape cen_weight to assign values for all halos
    cen_weight_all = np.concatenate(
        (cenpop.cen_weight, jnp.repeat(cenpop.cen_weight, subpop.nsub_per_host))
    )

    # Combine halo and subhalo mah_params
    mah_params_names = cenpop.mah_params._fields
    mah_params_tot = np.zeros((len(mah_params_names), n_host + n_sub))
    for i, _param in enumerate(mah_params_names):
        mah_params_tot[i, :] = np.concatenate(
            (
                cenpop.mah_params._asdict()[_param],
                subpop.mah_params._asdict()[_param],
            )
        )
    mah_params_all = namedtuple("mah_params", cenpop.mah_params._fields)(
        *mah_params_tot
    )

    # Reshape logmu_obs to assign values for all halos
    logmu_obs_all = jnp.concatenate((jnp.zeros(n_host), subpop.logmu_obs))

    # Reshape sat_weight to assign values for all halos
    sat_weight_all = jnp.concatenate((jnp.ones(n_host), subpop.sat_weight))

    # -- Create the output namedtuple containing host and subhalo information --
    values = (
        z_obs_all,
        t_obs_all,
        logmp_obs_all,
        mah_params_all,
        logmp0_all,
        cenpop.logt0,
        cen_weight_all,
        is_central,
        sat_weight_all,
        subpop.nsub_per_host,
        logmu_obs_all,
        halo_indx,
    )
    halopop = HaloPop(*values)

    return halopop


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

    """
    # two random keys, one for the host and one for the subhalo population
    cen_key, sub_key = jran.split(ran_key)

    # -- Compute cenpop: compute HMF and MAH params. for centrals. --
    cenpop = mclch.mc_lc_halos(
        cen_key,
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

    # -- Compute subpop: compute SHMF and MAH params. for subs. --
    subpop = mclcsh.mc_lc_subhalos(
        sub_key,
        cenpop,
        lgmsub_min,
        ccshmf_params=ccshmf_params,
        logmsub_cutoff=logmsub_cutoff,
        logmsub_cutoff_himass=logmsub_cutoff_himass,
        subhalo_model_key=subhalo_model_key,
    )

    halopop = _combine_cenpop_subpop(cenpop, subpop)

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

            cen_weight: ndarray of shape (n_halos_tot, )

                For centrals, cen_weight is determined by the halo mass function (HMF)
                For satellites, cen_weight is HMF weight of the associated central
                For satellites, cen_weight = halopop.cen_weight[halopop.halo_indx]

            central : ndarray of shape (n_halos_tot, )
                Integer equals 1 for central halos and 0 for subhalos

            sat_weight: ndarray of shape (n_halos_tot, )
                Multiplicity factor of the subhalo richness
                Equals 1 for central halos
                For subhalos, halopop.sat_weight = <Nsat(Msub) | Mhost>

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

    halopop = _combine_cenpop_subpop(cenpop, subpop)

    return halopop
