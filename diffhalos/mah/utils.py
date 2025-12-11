"""Useful utilities for diffmahpop"""

__all__ = ("rescale_mah_parameters",)


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
