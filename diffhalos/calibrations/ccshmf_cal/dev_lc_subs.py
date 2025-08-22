"""
Optimization tools for the conditional subhalo mass function
"""

from ...ccshmf import ccshmf_model as ccshmfm
from ...hmf.hmf_model import hmf_model, mc_hosts
from ...calibrations.fitting_utils.fitting_helpers import (
    jax_adam_wrapper,
)
from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad, vmap

N_GRID_HMF = 200
LGMP_GRID_HMF = jnp.linspace(0, 1, N_GRID_HMF)

N_GRID_CSHMF = 150
LGMU_UGRID = jnp.linspace(0, 1, N_GRID_CSHMF)

EPS = 1e-3

mult_matrix = vmap(
    vmap(lambda x, y: x * y, in_axes=(0, None)),
    in_axes=(None, 0),
)


@jjit
def _mean_subhalo_counts_kern(ccshmf_params, lgmhost, lgmp_min):
    """
    Predict the mean subhalo counts

    Parameters
    ----------
    ccshmf_params: namedtuple
        cumulative conditional subhalo mass function paramters

    lgmhost: float
        base-10 log of host halo mass

    lgmp_min: float
        base-10 log of the minimum particle mass

    Returns
    -------
    mean_counts: float
        predicted mean counts
    """
    lgmu_cutoff = lgmp_min - lgmhost
    mean_counts = 10 ** ccshmfm.predict_ccshmf(
        ccshmf_params,
        lgmhost,
        lgmu_cutoff,
    )
    return mean_counts


@jjit
def _predict_cuml_shmf_kern(
    hmf_params,
    ccshmf_params,
    redshift,
    lgmp_host_min,
    lgmp_host_max,
    lgmp_sub,
):
    lgmp_host_grid = lgmp_host_min + LGMP_GRID_HMF * (lgmp_host_max - lgmp_host_min)
    host_halo_mf = hmf_model.predict_differential_hmf(
        hmf_params, lgmp_host_grid, redshift
    )
    mean_counts_host_grid = _mean_subhalo_counts_kern(
        ccshmf_params, lgmp_host_grid, lgmp_sub
    )
    tot_counts = jnp.trapezoid(
        mean_counts_host_grid * host_halo_mf,
        lgmp_host_grid,
    )
    return tot_counts


_A = (*[None] * 6, 0)
predict_cuml_shmf_vmap = jjit(vmap(_predict_cuml_shmf_kern, in_axes=_A))


@jjit
def predict_cuml_shmf_from_ccshmf(
    hmf_params,
    ccshmf_params,
    redshift,
    lgmp_arr,
):
    lgmp_min = lgmp_arr.min()
    lgmp_host_min = lgmp_min + EPS
    lgmp_host_max = mc_hosts.LGMH_MAX
    return predict_cuml_shmf_vmap(
        hmf_params,
        ccshmf_params,
        redshift,
        lgmp_host_min,
        lgmp_host_max,
        lgmp_arr,
    )


@jjit
def _mae(pred, target):
    diff = pred - target
    return jnp.mean(jnp.abs(diff))


@jjit
def _loss_func_single_redshift(
    ccshmf_params,
    redshift,
    loss_data,
):
    """"""
    target_lgmparr = loss_data[1]
    target_host_halo_hmf_params = loss_data[2]
    target_subhalo_hmf_params = loss_data[3]

    target_log10_cuml_host_hmf = hmf_model.predict_cuml_hmf(
        target_host_halo_hmf_params, target_lgmparr, redshift
    )
    target_log10_cuml_hosts_and_subs_hmf = hmf_model.predict_cuml_hmf(
        target_subhalo_hmf_params, target_lgmparr, redshift
    )

    # Cumulative abundance of subhalos
    pred_cuml_shmf_subs = predict_cuml_shmf_from_ccshmf(
        target_host_halo_hmf_params, ccshmf_params, redshift, target_lgmparr
    )
    # Cumulative abundance of host halos
    pred_cuml_hmf_hosts = 10**target_log10_cuml_host_hmf

    # Cumulative abundance of (sub)halos
    pred_cuml_hosts_and_subs_hmf = pred_cuml_shmf_subs + pred_cuml_hmf_hosts

    pred_log10_cuml_hosts_and_subs_hmf = jnp.log10(
        pred_cuml_hosts_and_subs_hmf,
    )

    loss = _mae(
        pred_log10_cuml_hosts_and_subs_hmf,
        target_log10_cuml_hosts_and_subs_hmf,
    )
    return loss


_L = (None, 0, None)
_loss_func_multiz_kern = jjit(vmap(_loss_func_single_redshift, in_axes=_L))


@jjit
def loss_func_multiz(ccshmf_params, loss_data):
    zarr = loss_data[0]
    loss_arr = _loss_func_multiz_kern(ccshmf_params, zarr, loss_data)
    return jnp.mean(loss_arr)


_loss_and_grad_func = value_and_grad(loss_func_multiz, argnums=0)


def ccshmf_fitter(
    loss_data,
    p_init=ccshmfm.DEFAULT_CCSHMF_PARAMS,
    n_steps=200,
    step_size=0.01,
    n_warmup=1,
):
    _res = jax_adam_wrapper(
        _loss_and_grad_func,
        p_init,
        loss_data,
        n_steps,
        step_size=step_size,
        n_warmup=n_warmup,
    )
    p_best, loss, loss_hist, params_hist, fit_terminates = _res
    return p_best, loss, loss_hist, params_hist, fit_terminates
