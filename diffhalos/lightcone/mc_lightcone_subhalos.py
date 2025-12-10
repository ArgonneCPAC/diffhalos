# flake8: noqa: E402
"""Functions to generate subhalo lightcones"""

from jax import config

config.update("jax_enable_x64", True)

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


def mc_lightcone_subhalo_mass_function(
    ran_key,
    lgmhost_arr,
    lgmp_min,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
):
    mc_lg_mu, lgmhost_pop, host_halo_indx = generate_subhalopop(
        ran_key,
        lgmhost_arr,
        lgmp_min,
        ccshmf_params=ccshmf_params,
    )

    return mc_lg_mu, lgmhost_pop, host_halo_indx


def mc_lightcone_subhalo_diffmah():

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
        with keys:
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
        CCSHMF parameters named tuple

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
