"""Useful utilities for mah related computations"""

from functools import partial

from jax import jit as jjit

from ..mah.diffmahnet.diffmahnet import log_mah_kern
from ..mah.diffmahnet_utils import mc_mah_cenpop

__all__ = ("rescale_mah_parameters",)


@jjit
def rescale_mah_parameters(
    mah_params_uncorrected,
    logm_obs,
    logm_obs_uncorrected,
):
    """
    Corrects the mah model parameters, so that
    logm0 is rescaled to the value that results in
    mah's that agree with the observed halo mass

    Parameters
    ----------
    mah_params_uncorrected: namedtuple
        mah parameters (logm0, logtc, early_index, late_index, t_peak)
        where each parameters is a ndarray of shape (n_halo, )

    logm_obs: ndarray of shape (n_halo, )
        base-10 log of true observed halo masses, in Msun

    logm_obs: ndarray of shape (n_halo, )
        base-10 log of uncorrected observed halo masses, in Msun

    Returns
    -------
    mah_params: namedtuple
        mah parameters after rescaling, of same shape as ``mah_uncorrected``
    """
    delta_logm_obs = logm_obs_uncorrected - logm_obs
    logm0_rescaled = mah_params_uncorrected.logm0 - delta_logm_obs
    mah_params = mah_params_uncorrected._replace(logm0=logm0_rescaled)

    return mah_params


@partial(jjit, static_argnames=["centrals_model_key"])
def apply_mah_rescaling(
    mah_key,
    logmp_obs_mf,
    logmp_obs_clipped,
    t_obs,
    logt0,
    centrals_model_key,
):

    mah_params_uncorrected = mc_mah_cenpop(
        logmp_obs_clipped,
        t_obs,
        mah_key,
        centrals_model_key,
    )

    # compute the uncorrected observed masses
    logmp_obs_uncorrected = log_mah_kern(mah_params_uncorrected, t_obs, logt0)

    # rescale the mah parameters to the correct logm0
    mah_params = rescale_mah_parameters(
        mah_params_uncorrected,
        logmp_obs_mf,
        logmp_obs_uncorrected,
    )

    # compute observed mass with corrected parameters
    logmp_obs = log_mah_kern(mah_params, t_obs, logt0)

    return mah_params, logmp_obs
