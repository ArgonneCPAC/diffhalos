# flake8: noqa: E402
"""Functions to generate subhalo lightcones"""

from jax import config

config.update("jax_enable_x64", True)

from jax import jit as jjit
from jax import numpy as jnp

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

__all__ = (
    "mc_lc_shmf",
    "mc_lc_subhalos",
)

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
            mah_params: namedtuple of ndarray's with shape (n_subs, n_mah_params)
                diffmah parameters for each subhalo in the lightcone
    """

    # generate a Poisson realization of subhalos, given the host halo population
    mc_lg_mu_shmf, _, host_index_for_sub, n_mu_per_host = generate_subhalopop(
        ran_key,
        halopop.logmp_obs,
        lgmp_min,
        ccshmf_params=ccshmf_params,
    )

    # get the subhalo mass and time of observation for MAH computations
    logmsub_obs_shmf = jnp.repeat(halopop.logmp_obs, n_mu_per_host) + mc_lg_mu_shmf
    t_obs = jnp.repeat(halopop.t_obs, n_mu_per_host)

    # get the uncorrected MAH parameters for all subhalos
    logmsub_obs_clipped = jnp.clip(
        logmsub_obs_shmf, logmsub_cutoff, logmsub_cutoff_himass
    )
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
        logmsub_obs_shmf,
        logmsub_obs_uncorrected,
    )

    # compute observed mass and mu values with corrected parameters
    logmsub_obs = log_mah_kern(mah_params_subs, t_obs, logt0)
    mc_lg_mu = logmsub_obs - halopop.logmp_obs

    # add the subhalo data to the halo population namedtuple
    new_fields = ("host_index_for_sub", "mah_params_sub", "logmu_subs")
    new_data = (host_index_for_sub, mah_params_subs, mc_lg_mu)
    halopop = add_field_to_namedtuple(halopop, new_fields, new_data)

    return halopop


@partial(jjit, static_argnames=("subhalo_model_key", "n_mu_per_host"))
def weighted_subhalo_lightcone(
    halopop,
    ran_key,
    lgmp_min,
    lgt0,
    n_mu_per_host=N_LGMU_PER_HOST,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
    subhalo_model_key=DEFAULT_DIFFMAHNET_SAT_MODEL,
):
    """
    Generate a subhalo lightcone

    Parameters
    ----------
    halopop: dict
        halo population, with necessary keys:
            logmp_obs: ndarray of shape (n_host, )
                base-10 log of halo mass at observation, in Msun

            t_obs: ndarray of shape (n_host, )
                cosmic time at observation, in Gyr

    lgmp_min: float
        base-10 log of the minimum mass, in Msun

    lgt0: float
        base-10 log of cosmic time today, in Gyr

    n_mu_per_host: int
        number of mu=Msub/Mhost values to use per host halo;
        note that for the weighted version of the lightcone,
        each host gets assigned the same number of subhalos

    cshmf_params: namedtuple
        CCSHMF parameters

    subhalo_model_key: str
        diffmahnet model to use for satellites

    Returns
    -------
    halopop: dict
        same as input with added keys:
            nsubhalos: ndarray of shape (n_host*n_mu, )
                subhalo number weights

            logmu_subs: ndarray of shape (n_host*n_mu, )
                base-10 log of mu values per host halo
    """

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

    # generate diffmah subhalo populations
    n_mu_per_host_arr = jnp.repeat(jnp.asarray(n_mu_per_host), n_host)
    halopop = _mc_lightcone_subhalo(
        ran_key,
        halopop,
        lgt0,
        lgmu,
        n_mu_per_host_arr,
        int(n_mu_per_host * n_host),
        subhalo_model_key=subhalo_model_key,
    )

    # add subhalo weights to the dictionary
    new_fields = ("nsubhalos", "logmu_subs")
    new_data = (nsubhalo_weights, lgmu)
    halopop = add_field_to_namedtuple(halopop, new_fields, new_data)

    return halopop
