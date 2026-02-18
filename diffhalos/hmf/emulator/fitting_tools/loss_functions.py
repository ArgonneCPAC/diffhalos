"""Useful utilities for fitting"""

from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap

from ...hmf_model_flat import wrapper_ndarray_diffsky_diff_hmf


__all__ = ("mse", "mse_loss_hmf_params", "mse_loss_diff_hmf_curve")


@jjit
def mse(pred, target):
    """
    Mean squared error loss function

    Parameters
    ----------
    pred: ndarray of shape (n_data, )
        predicted values by the model

    target: ndarray of shape (n_data, )
        target data

    Returns
    -------
    mse_val: float
        mse value
    """
    diff = pred - target
    mse_val = jnp.mean(diff**2)

    return mse_val


@jjit
def mse_loss_hmf_params(preds, target_data):
    """
    Mean squared error loss function
    on the halo mass function parameters directly

    Parameters
    ----------
    preds: ndarray of shape (n_cosmo, n_hmf_params)
        predicted HMF parameters by the model

    target_data: ndarray of shape (n_cosmo, n_hmf_params)
        target HMF parameters

    Returns
    -------
    mse_val: float
        mean squared error value between
        nn's prediction and target data
    """

    mse_val = mse(preds, target_data)

    return mse_val


@jjit
def mse_loss_diff_hmf_curve(
    preds,
    target_data,
    logmp,
    z,
):
    """
    Mean squared error loss function
    on the halo mass funcion curve

    Parameters
    ----------
    preds: ndarray of shape (n_cosmo, n_hmf_params)
        predicted HMF parameters by the model

    target_data: ndarray of shape (n_cosmo, n_redshift, n_halo)
        target data for loss calculation
        against current optimizer's state;
        in this case the target HMF parameters

    logmp: ndarray of shape (n_cosmo, n_redshift, n_halo)
        base-10 log of halo mass per cosmology and redshift

    z: ndarray of shape (n_redshift, )
        redshift values considered

    Returns
    -------
    mse: float
        mean squared error value between
        nn's prediction and target data
    """
    mse_val = 0.0
    for i, zi in enumerate(z):
        loghmf_preds = _get_diff_hmf_vmap(preds, logmp[:, i, :], zi)
        mse_val += mse(loghmf_preds, target_data[:, i, :])

    return mse_val


_get_diff_hmf_vmap = jjit(
    vmap(
        wrapper_ndarray_diffsky_diff_hmf,
        in_axes=(0, 0, None),
    )
)
