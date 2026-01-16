# flake8: noqa: E402
"""Functions to generate subhalo lightcones"""

from jax import config

config.update("jax_enable_x64", True)

from jax import jit as jjit
from jax import numpy as jnp

from functools import partial

from ..mah.diffmahnet_utils import mc_mah_satpop as mc_mah_satpop_diffmahnet
from ..ccshmf.mc_subs import generate_subhalopop
from ..ccshmf.ccshmf_model import (
    subhalo_lightcone_weights,
    DEFAULT_CCSHMF_PARAMS,
)
from ..utils.namedtuple_utils import add_field_to_namedtuple

__all__ = (
    "mc_lightcone_subhalo_mass_function",
    "mc_lightcone_subhalo_diffmah",
    "mc_weighted_subhalo_lightcone",
)

DEFAULT_DIFFMAHNET_SAT_MODEL = "satflow_v2_0_64bit.eqx"
N_LGMU_PER_HOST = 5


@partial(jjit, static_argnames=("nsub_tot",))
def mc_lightcone_subhalo_mass_function(
    ran_key,
    lgmhost,
    lgmp_min,
    subhalo_counts_per_halo,
    nsub_tot,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
):
    """
    Generate MC generalization of the subhalo mass function in a lightcone

    Parameters
    ----------
    ran_key: jran.key
        random key

    lgmhost: ndarray of shape (n_host, )
        base-10 log of host halo mass, in Msun

    lgmp_min: float
        base-10 log of the minimum subhalo mass, in Msub

    subhalo_counts_per_halo: ndarray of shape (nsubs, )
        subhalo counts per host halo;
        note that the total, i.e. subhalo_counts_per_halo.sum(),
        must be the same as ``nsub_tot``, and thus it should
        be nsub_tot=subhalo_counts_per_halo.sum()

    nsub_tot: int
        number of subhalos to generate

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
    """
    mc_lg_mu, lgmhost_pop, host_halo_indx = generate_subhalopop(
        ran_key,
        lgmhost,
        lgmp_min,
        subhalo_counts_per_halo,
        nsub_tot,
        ccshmf_params=ccshmf_params,
    )

    return mc_lg_mu, lgmhost_pop, host_halo_indx


@partial(jjit, static_argnames=("nsub_tot", "subhalo_model_key"))
def mc_lightcone_subhalo_diffmah(
    ran_key,
    halopop,
    lgt0,
    lgmu,
    n_mu_per_host,
    nsub_tot,
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
        halo population, with necessary fields:
            logmp_obs: ndarray of shape (n_host, )
                base-10 log of halo mass at observation, in Msun

            t_obs: ndarray of shape (n_host, )
                cosmic time at observation, in Gyr

    lgt0: float
        base-10 log of cosmic time today, in Gyr

    lgmu: ndarray of shape (n_sub, )
        values of mu=Msub/Mhost for all subhalos in the lightcone

    n_mu_per_host: float
        number of mu=Msub/Mhost values to use per host halo

    nsub_tot: int
        number of subhalos to generate;
        note that it must be nsub_tot=n_mu_per_host.sum()

    subhalo_model_key: str
        diffmahnet model to use for satellites

    Returns
    -------
    halopop: dict
        same as input with added keys:
            mah_params: namedtuple of ndarray's with shape (n_subs, n_mah_params)
                diffmah parameters for each subhalo in the lightcone
    """

    n_host = halopop.logmp_obs.size
    n_sub = lgmu.size

    # match host index with subhalos
    host_index_for_sub = jnp.repeat(
        jnp.arange(n_host),
        n_mu_per_host,
        total_repeat_length=nsub_tot,
    )

    # get the MAH parameters for the subhalos
    tarr = (jnp.ones(n_sub) * lgt0).reshape(n_sub, 1)
    _, mah_params_sub = mc_mah_satpop_diffmahnet(
        jnp.repeat(
            halopop.logmp_obs,
            n_mu_per_host,
            total_repeat_length=nsub_tot,
        )
        * lgmu,
        jnp.repeat(
            halopop.t_obs,
            n_mu_per_host,
            total_repeat_length=nsub_tot,
        ),
        ran_key,
        tarr,
        subhalo_model_key=subhalo_model_key,
        logt0=lgt0,
    )

    new_fields = ("host_index_for_sub", "mah_params_sub")
    new_data = (host_index_for_sub, mah_params_sub)
    halopop = add_field_to_namedtuple(halopop, new_fields, new_data)

    return halopop


@partial(jjit, static_argnames=("subhalo_model_key", "n_mu_per_host"))
def mc_weighted_subhalo_lightcone(
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
    halopop = mc_lightcone_subhalo_diffmah(
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
