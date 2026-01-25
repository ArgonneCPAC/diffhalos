# flake8: noqa: E402
"""Functions to generate subhalo lightcones"""

from jax import config

config.update("jax_enable_x64", True)

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from functools import partial

from ..mah.diffmahnet_utils import mc_mah_satpop
from ..ccshmf.mc_subs import generate_subhalopop
from ..ccshmf.ccshmf_model import (
    subhalo_lightcone_weights,
    DEFAULT_CCSHMF_PARAMS,
)
from ..mah.diffmahnet.diffmahnet import log_mah_kern
from ..mah.utils import rescale_mah_parameters
from ..utils.namedtuple_utils import add_field_to_namedtuple

__all__ = ("mc_lc_shmf", "mc_lc_subhalos", "weighted_lc_subhalos")

DEFAULT_DIFFMAHNET_SAT_MODEL = "satflow_v2_0_64bit.eqx"
N_LGMU_PER_HOST = 5


DEFAULT_LOGMSUB_CUTOFF = 10.0
DEFAULT_LOGMSUB_HIMASS_CUTOFF = 14.5


def mc_lc_shmf(
    ran_key,
    lgmhost_arr,
    lgmp_min,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
):
    """
    Generate MC generalization of the subhalo mass function in a lightcone,
    provided the masses of their host halos.
    This function is essentially a convenient wrapper around the
    ``mc_subs.generate_subhalopop`` function

    Parameters
    ----------
    ran_key: jran.key
        random key

    lgmhost_arr: ndarray of shape (n_host, )
        base-10 log of host halo mass, in Msun

    lgmp_min: float
        base-10 log of the minimum subhalo mass, in Msub

    ccshmf_params: namedtuple
        CCSHMF parameters

    Returns
    -------
    mc_lg_mu: ndarray of shape (n_mu, )
        base-10 log of mu=Msub/Mhost of the Monte Carlo subhalo population

    lgmhost_pop: ndarray of shape (n_host, )
        base-10 log of Mhost of the Monte Carlo subhalo population, in Msun

    host_halo_indx: ndarray of shape (n_mu*n_host, )
        index of the input host halo of each generated subhalo,
        so that lgmhost_pop = lgmhost_arr[host_halo_indx];
        thus all values satisfy 0 <= host_halo_indx < nhosts

    subhalo_counts_per_halo: ndarray of shape (m_hosts, )
        number of generated subhalos per host halo
    """
    mc_lg_mu, lgmhost_pop, host_halo_indx, subhalo_counts_per_halo = (
        generate_subhalopop(
            ran_key,
            lgmhost_arr,
            lgmp_min,
            ccshmf_params=ccshmf_params,
        )
    )

    return mc_lg_mu, lgmhost_pop, host_halo_indx, subhalo_counts_per_halo


def mc_lc_subhalos(
    ran_key,
    halopop,
    lgmp_min,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
    logmsub_cutoff=DEFAULT_LOGMSUB_CUTOFF,
    logmsub_cutoff_himass=DEFAULT_LOGMSUB_HIMASS_CUTOFF,
    subhalo_model_key=DEFAULT_DIFFMAHNET_SAT_MODEL,
):
    """
    Computes MAHs for subhalo populations,
    based on the ``diffmahnet`` model

    Parameters
    ----------
    ran_key: jran.key
        random key

    halopop: namedtuple
        central population, with necessary fields:
            logmp_obs: ndarray of shape (n_host, )
                base-10 log of halo mass at observation, in Msun

            t_obs: ndarray of shape (n_host, )
                cosmic time at observation, in Gyr

            logt0: float
                base-10 log of cosmic time at today, in Gyr

    lgmp_min: float
        absolute minimum subhalo mass, in Msun

    ccshmf_params: namedtuple
        CCSHMF parameters

    logmsub_cutoff: float
        base-10 log of minimum subhalo mass for which
        diffmahnet is used to generate MAHs, in Msun;
        for logmsub < logmsub_cutoff, P(θ_MAH | logmsub) = P(θ_MAH | logmsub_cutoff)

    logmsub_cutoff_himass: float
        base-10 log of maximum subhalo mass for which
        diffmahnet is used to generate MAHs, in Msun

    subhalo_model_key: str
        diffmahnet model to use for satellites

    Returns
    -------
    halopop: namedtuple
        same as input ``halopop`` with added keys:
            host_index_for_subs: ndarray of shape (n_subs_per_host*n_host, )
                index of the host halo of each generated subhalo

            mah_params_subs: namedtuple of ndarray's with shape (n_subs, n_mah_params)
                diffmah parameters for each subhalo in the lightcone

            logmu_obs: ndarray of shape (n_subs, )
                base-10 log of mu=Msub/Mhost for each subhalo in the lightcone
    """

    # two random keys, one for the MC subhalo population and one for diffmahnet
    mu_key, mah_key = jran.split(ran_key, 2)

    # generate a Poisson realization of subhalos, given the host halo population
    mc_lg_mu_shmf, _, host_index_for_subs, n_mu_per_host = generate_subhalopop(
        mu_key,
        halopop.logmp_obs,
        lgmp_min,
        ccshmf_params=ccshmf_params,
    )

    # get the subhalo mass and time of observation for MAH computations
    logmsub_obs_shmf = mc_lg_mu_shmf + jnp.repeat(halopop.logmp_obs, n_mu_per_host)
    t_obs = jnp.repeat(halopop.t_obs, n_mu_per_host)

    # get the uncorrected MAH parameters for all subhalos
    logmsub_obs_clipped = jnp.clip(
        logmsub_obs_shmf, logmsub_cutoff, logmsub_cutoff_himass
    )
    mah_params_subs_uncorrected = mc_mah_satpop(
        logmsub_obs_clipped,
        t_obs,
        mah_key,
        subhalo_model_key,
    )

    # compute the uncorrected observed masses
    logt0 = halopop.logt0
    logmsub_obs_uncorrected = log_mah_kern(mah_params_subs_uncorrected, t_obs, logt0)

    # rescale the mah parameters to the correct logm0
    mah_params_subs = rescale_mah_parameters(
        mah_params_subs_uncorrected,
        logmsub_obs_shmf,
        logmsub_obs_uncorrected,
    )

    # compute observed mass and mu values with corrected parameters
    logmsub_obs = log_mah_kern(mah_params_subs, t_obs, logt0)
    mc_lg_mu = logmsub_obs - jnp.repeat(halopop.logmp_obs, n_mu_per_host)

    # add the subhalo data to the halo population namedtuple
    new_fields = ("host_index_for_subs", "mah_params_subs", "logmu_obs")
    new_data = (host_index_for_subs, mah_params_subs, mc_lg_mu)
    halopop = add_field_to_namedtuple(halopop, new_fields, new_data)

    return halopop


@partial(jjit, static_argnames=("subhalo_model_key", "n_mu_per_host"))
def weighted_lc_subhalos(
    halopop,
    ran_key,
    lgmp_min,
    n_mu_per_host=N_LGMU_PER_HOST,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
    logmsub_cutoff=DEFAULT_LOGMSUB_CUTOFF,
    logmsub_cutoff_himass=DEFAULT_LOGMSUB_HIMASS_CUTOFF,
    subhalo_model_key=DEFAULT_DIFFMAHNET_SAT_MODEL,
):
    """
    Generate a subhalo lightcone

    Parameters
    ----------
    halopop: namedtuple
        central population, with necessary fields:
            logmp_obs: ndarray of shape (n_host, )
                base-10 log of halo mass at observation, in Msun

            t_obs: ndarray of shape (n_host, )
                cosmic time at observation, in Gyr

            logt0: float
                base-10 log of cosmic time at today, in Gyr

    lgmp_min: float
        base-10 log of the minimum mass, in Msun

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

    subhalo_model_key: str
        diffmahnet model to use for satellites

    Returns
    -------
    halopop: dict
        same as input with added keys:
            host_index_for_subs: ndarray of shape (n_subs_per_host*n_host, )
                index of the host halo of each generated subhalo

            mah_params_subs: namedtuple of ndarray's with shape (n_subs, n_mah_params)
                diffmah parameters for each subhalo in the lightcone

            logmu_obs: ndarray of shape (n_subs, )
                base-10 log of mu=Msub/Mhost for each subhalo in the lightcone

            nsubhalos: ndarray of shape (n_nub, )
                subhalo weighted counts
    """

    # number of host halos
    n_host = halopop.logmp_obs.size

    # get subhalo weights
    nsubhalo_weights, lgmu = subhalo_lightcone_weights(
        halopop.logmp_obs,
        lgmp_min,
        n_mu_per_host,
        ccshmf_params,
    )
    nsubhalo_weights = nsubhalo_weights.reshape(n_host * n_mu_per_host)
    lgmu = lgmu.reshape(n_host * n_mu_per_host)

    # get the subhalo mass and time of observation for MAH computations
    logmsub_obs = lgmu + jnp.repeat(halopop.logmp_obs, n_mu_per_host)
    t_obs = jnp.repeat(halopop.t_obs, n_mu_per_host)

    # get the uncorrected MAH parameters for all subhalos
    logmsub_obs_clipped = jnp.clip(logmsub_obs, logmsub_cutoff, logmsub_cutoff_himass)
    mah_params_subs_uncorrected = mc_mah_satpop(
        logmsub_obs_clipped,
        t_obs,
        ran_key,
        subhalo_model_key,
    )

    # compute the uncorrected observed masses
    logt0 = halopop.logt0
    logmsub_obs_uncorrected = log_mah_kern(mah_params_subs_uncorrected, t_obs, logt0)

    # rescale the mah parameters to the correct logm0
    mah_params_subs = rescale_mah_parameters(
        mah_params_subs_uncorrected,
        logmsub_obs,
        logmsub_obs_uncorrected,
    )

    # compute observed mass and mu values with corrected parameters
    logmsub_obs = log_mah_kern(mah_params_subs, t_obs, logt0)
    logmu_obs = logmsub_obs - jnp.repeat(halopop.logmp_obs, n_mu_per_host)

    # construct the host index array for the subhalos
    halo_ids = jnp.arange(halopop.logmp_obs.size).astype(int)
    host_index_for_subs = jnp.repeat(halo_ids, n_mu_per_host)

    # add subhalo weights to the dictionary
    new_fields = ("nsubhalos", "host_index_for_subs", "mah_params_subs", "logmu_obs")
    new_data = (nsubhalo_weights, host_index_for_subs, mah_params_subs, logmu_obs)
    halopop = add_field_to_namedtuple(halopop, new_fields, new_data)

    return halopop
