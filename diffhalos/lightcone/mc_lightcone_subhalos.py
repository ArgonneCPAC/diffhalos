# flake8: noqa: E402
"""Functions to generate subhalo lightcones"""

from jax import config

config.update("jax_enable_x64", True)

import numpy as np

from ..mah.diffmahnet_utils import mc_mah_satpop as mc_mah_satpop_diffmahnet
from diffmah.diffmah_kernels import _log_mah_kern

from ..ccshmf.ccshmf_model import N_LGMU_TABLE  # noqa
from ..ccshmf.mc_subs import generate_subhalopop
from ..ccshmf.ccshmf_model import (
    subhalo_lightcone_weights,
    DEFAULT_CCSHMF_PARAMS,
)

__all__ = (
    "mc_lightcone_subhalo_mass_function",
    "mc_lightcone_subhalo_diffmah",
    "mc_weighted_subhalo_lightcone",
)

DEFAULT_DIFFMAHNET_SAT_MODEL = "satflow_v1_0train_64bit.eqx"


def mc_lightcone_subhalo_mass_function(
    ran_key,
    lgmhost_arr,
    lgmp_min,
    ccshmf_params=DEFAULT_DIFFMAHNET_SAT_MODEL,
):
    mc_lg_mu, lgmhost_pop, host_halo_indx = generate_subhalopop(
        ran_key,
        lgmhost_arr,
        lgmp_min,
        ccshmf_params=ccshmf_params,
    )

    return mc_lg_mu, lgmhost_pop, host_halo_indx


def mc_lightcone_subhalo_diffmah(
    ran_key,
    logm_obs,
    t_obs,
    lgt0,
    subhalo_model_key=DEFAULT_DIFFMAHNET_SAT_MODEL,
):
    """
    Computes MAHs for subhalo populations,
    based on the ``diffmahnet`` model

    Parameters
    ----------
    ran_key: jran.key
        random key

    logm_obs: ndarray of shape (n_subs, )
        base-10 log of subhalo mass, in Msun

    t_obs: ndarray of shape (n_subs, )
        cosmic time at observation, in Gyr

    lgt0: float
        base-10 log of cosmic time today, in Gyr

    subhalo_model_key: str
        diffmahnet model to use for satellites

    Returns
    -------
    """

    # get the MAH parameters for the halos
    num_halos = t_obs.size
    tarr = (np.ones(num_halos) * lgt0).reshape(num_halos, 1)
    logm_obs, mah_params = mc_mah_satpop_diffmahnet(
        logm_obs,
        t_obs,
        ran_key,
        tarr,
        subhalo_model_key=subhalo_model_key,
        logt0=lgt0,
    )
    logm_obs = np.concatenate(logm_obs)

    # compute MAH values today
    logm0 = _log_mah_kern(mah_params, 10**lgt0, lgt0)

    return


def mc_weighted_subhalo_lightcone(
    halopop,
    lgmp_min,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
):
    """
    Generate a subhalo lightcone

    Parameters
    ----------
    halopop: dict
        halo population, with keys:
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

    lgmp_min: float
        base-10 log of the minimum mass, in Msun

    cshmf_params: namedtuple
        CCSHMF parameters

    Returns
    -------
    halopop: dict
        same as input with the added keys:
            nsubhalos: ndarray of shape (n_halo, N_LGMU_TABLE)
                subhalo number weights

            logmu_subs: ndarray of shape (n_halo, N_LGMU_TABLE)
                base-10 log of mu values per host halo
    """

    # get subhalo weights
    nsubhalo_weights, lgmu = subhalo_lightcone_weights(
        halopop["logmp_obs"],
        lgmp_min,
        ccshmf_params,
    )

    # add subhalo weights to the dictionary
    halopop["nsubhalos"] = nsubhalo_weights
    halopop["logmu_subs"] = lgmu

    return halopop
