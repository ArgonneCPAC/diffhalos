"""Fitting tools for halo mass function"""

from .hmf_diff_fitter import diff_hmf_fitter
from .hmf_cuml_fitter import cuml_hmf_fitter
from ...hmf_param_utils import DEFAULT_HMF_PARAMS as P_INIT


__all__ = ("fit_hmf_single_cosmo", "fit_hmf_multi_cosmo")


def fit_hmf_single_cosmo(
    loss_data,
    n_steps=2000,
    step_size=0.01,
    n_warmup=1,
    cuml=False,
    p_init=P_INIT,
):
    """
    Fit a model to either the cumulative or differential
    halo mass function, at single cosmology,
    using jax's Adam, following the implementation in
    ``diffsky.mass_functions.fitting_utils.fit_hmf_model``;
    all redshifts are being fitted together, so we have at the end
    a set of best-fit HMF model parameters for the entire
    input redshift range

    Parameters
    ----------
    loss_data: list
        each element is a list of
        [redshift,
         base-10 log of halo mass,
         base-10 log of cumulative halo mass function]

    n_steps: int
        number of steps to take

    step_size: float
        step size

    n_warmup: int
        number of warmup steps

    cuml: bool
        if True, will run fitter for the cumulative HMF,
        if False, will run fitter for the differential HMF

    p_init: namedtuple
        initial guess of HMF parameters

    Returns
    -------
    res: tuple
        tuple containing the following

        p_best: namedtuple
            ``HMF_Params`` namedtuple with the
            best-fit diffsky HMF parameters

        loss: float
            loss at the last step

        loss_hist: ndarray of shape (n_steps,)
            loss at each step

        params_hist: list of length ``n_steps``
            HMF parameters at each step

        fit_terminates:
            0 if NaN or inf is encountered by the fitter,
            causing termination before n_step, or
            1 for a fit that terminates with no such problems
    """
    if cuml:
        res = cuml_hmf_fitter(
            loss_data,
            n_steps=n_steps,
            step_size=step_size,
            n_warmup=n_warmup,
            p_init=p_init,
        )
    else:
        res = diff_hmf_fitter(
            loss_data,
            n_steps=n_steps,
            step_size=step_size,
            n_warmup=n_warmup,
            p_init=p_init,
        )

    return res


def fit_hmf_multi_cosmo(
    loss_data,
    n_steps=2000,
    step_size=0.01,
    n_warmup=1,
    cuml=False,
    p_init=P_INIT,
):
    """
    Fit a model to either the cumulative or differential
    halo mass function, at multiple cosmologies,
    using the single-cosmology fitter
    ``.fit_hmf_single_cosmo``;
    at each cosmology, all redshifts are being fitted together
    and in the end we have best-fit HMF model parameters
    for each cosmology over the entire input redshift range

    Parameters
    ----------
    loss_data: list
        this is a list of lists with structure
        [redshift,
         base-10 log of halo mass,
         base-10 log of cumulative halo mass function]

    n_steps: int
        number of steps to take

    step_size: float
        step size

    n_warmup: int
        number of warmup steps

    cuml: bool
        if True, will run fitter for the cumulative HMF,
        if False, will run fitter for the differential HMF

    p_init: namedtuple
        initial guess of HMF parameters

    Returns
    -------
    result_collector: list
        contains the results from all cosmology fits
        as tuples with the following

        p_best: namedtuple
            ``HMF_Params`` namedtuple with the
            best-fit diffsky HMF parameters

        loss: float
            loss at the last step

        loss_hist: ndarray of shape (n_steps,)
            loss at each step

        params_hist: list of length ``n_steps``
            HMF parameters at each step

        fit_terminates:
            0 if NaN or inf is encountered by the fitter,
            causing termination before n_step, or
            1 for a fit that terminates with no such problems
    """

    result_collector = []
    for _loss_data in loss_data:
        if cuml:
            res = cuml_hmf_fitter(
                _loss_data,
                n_steps=n_steps,
                step_size=step_size,
                n_warmup=n_warmup,
                p_init=p_init,
            )
        else:
            res = diff_hmf_fitter(
                _loss_data,
                n_steps=n_steps,
                step_size=step_size,
                n_warmup=n_warmup,
                p_init=p_init,
            )

        result_collector.append(res)

    return result_collector
