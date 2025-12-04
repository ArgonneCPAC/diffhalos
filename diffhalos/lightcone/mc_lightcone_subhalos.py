"""Functions to generate Monte Carlo realizations of subhalos in a lightcone"""

from ..ccshmf.ccshmf_model import (
    DEFAULT_CCSHMF_PARAMS,
    subhalo_lightcone_weights,
)

__all__ = ("mc_weighted_subhalo_lightcone",)


def mc_weighted_subhalo_lightcone(
    ran_key,
    halopop,
    lgmu,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
):
    """
    Generate a subhalo lightcone

    Parameters
    ----------
    ran_key: jax.random.PRNGKey
        random key

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

    lgmu: ndarray of shape (n_mu, )
        base-10 log of mu=Msub/Mhost values to consider

    cshmf_params: namedtuple
        CCSHMF parameters named tuple

    Returns
    -------
    halopop: dict
        same as input with the added key
        ``nsubhalos`` for the subhalo number weights
    """

    # get subhalo weights
    nsubhalo_weights = subhalo_lightcone_weights(
        ran_key,
        halopop["logmp_obs"],
        lgmu,
        ccshmf_params=ccshmf_params,
    )

    # add subhalo weights to the dictionary
    halopop["nsubhalos"] = nsubhalo_weights

    return halopop
