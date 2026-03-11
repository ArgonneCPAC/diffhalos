"""Helper fitting tools for differential halo mass function fits"""

from jax import jit as jjit
from jax import value_and_grad

from .loss_functions import mse
from .fitting_helpers import jax_adam_wrapper
from ...hmf_param_utils import HMF_Params
from ...hmf_param_utils import DEFAULT_HMF_PARAMS as P_INIT
from ...hmf_model import predict_diff_hmf

__all__ = ("diff_hmf_fitter",)


def diff_hmf_fitter(
    loss_data,
    n_steps=200,
    step_size=0.01,
    n_warmup=1,
    p_init=P_INIT,
):
    """
    Runs a fitter to the differential diffsky halo mass function

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

    p_init: namedtuple
        initial guess of HMF parameters

    Returns
    -------
    p_best: namedtuple
        `HMF_Params` named tuple with the
        best-fit diffsky HMF parameters

    loss: float
        loss at the last step

    loss_hist: ndarray of shape (n_steps, )
        loss at each step

    params_hist: list of length `n_steps`
        HMF parameters at each step

    fit_terminates:
        0 if NaN or inf is encountered by the fitter,
        causing termination before n_step, or
        1 for a fit that terminates with no such problems
    """
    _res = jax_adam_wrapper(
        _loss_and_grad_func,
        p_init,
        loss_data,
        n_steps,
        step_size=step_size,
        n_warmup=n_warmup,
    )
    p_best, loss, loss_hist, params_hist, fit_terminates = _res
    p_best = HMF_Params(*p_best)
    return p_best, loss, loss_hist, params_hist, fit_terminates


@jjit
def _loss_func_single_redshift(params, loss_data):
    """computes loss at a single redshift point"""
    redshift, target_lgmp, target_hmf = loss_data
    pred_hmf = predict_diff_hmf(params, target_lgmp, redshift)
    loss = mse(pred_hmf, target_hmf)
    return loss


@jjit
def _loss_func_multi_z(params, loss_data):
    """computes loss at multiple redshift points"""
    loss = 0.0
    for single_z_data in loss_data:
        loss = loss + _loss_func_single_redshift(params, single_z_data)
    return loss


"""computes loss and gradient at multiple redshift points"""
_loss_and_grad_func = jjit(value_and_grad(_loss_func_multi_z, argnums=0))
