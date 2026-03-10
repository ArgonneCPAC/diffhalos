"""Useful utilities for fitting"""

from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap

from ...hmf_model import predict_diff_hmf, predict_cuml_hmf
from ...hmf_param_utils import define_diffsky_hmf_params_namedtuple

__all__ = ("mse", "mse_loss_diff_hmf_curve", "mse_loss_cuml_hmf_curve")


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
def mse_hmf_params(pred, target):
    """
    Loss function comparing directly the
    mlp-predicted and target hmf parameters

    Parameters
    ----------
    pred: ndarray of shape (n_cosmo, n_hmf_params)
        predicted hmf parameters per cosmology sample

    target: ndarray of shape (n_cosmo, n_hmf_params)
        target hmf parameters per cosmology sample

    Returns
    -------
    mse_val: float
        mse value
    """
    diff = (pred - target) ** 2
    mse_val = jnp.sum(jnp.mean(diff, axis=1))

    return mse_val


@jjit
def mse_loss_diff_hmf_curve(preds, target, logmp, z):
    """
    Mean squared error loss function based
    on the differential halo mass funcion curve

    Parameters
    ----------
    pred: ndarray of shape (n_cosmo, n_hmf_params)
        predicted hmf parameters per cosmology sample

    target: ndarray of shape (n_cosmo, n_redshift, n_halo)
        target hmf curves per cosmology sample and redshift

    logmp: ndarray of shape (n_cosmo, n_redshift, n_halo)
        halo masses per cosmology sample and redshift, in Msun

    z: ndarray of shape (n_redshift, )
        redshift values for loss computation

    Returns
    -------
    mse: float
        mean squared error value
    """

    mse_val = 0.0
    for i in range(target.shape[0]):
        preds_ntup = define_diffsky_hmf_params_namedtuple(preds[i])
        loghmf_preds = _predict_diff_hmf_vmap(preds_ntup, logmp[i], z)
        diff = (loghmf_preds - target[i]) ** 2
        mse_val += jnp.sum(jnp.mean(diff, axis=1))

    return mse_val


_predict_diff_hmf_vmap = jjit(vmap(predict_diff_hmf, in_axes=(None, 0, 0)))


@jjit
def mse_loss_cuml_hmf_curve(preds, target_data):
    """
    Mean squared error loss function based
    on the cumulative halo mass funcion curve

    Parameters
    ----------
    preds: ndarray of shape (n_cosmo, n_hmf_params)
        prediction for hmf parameters by neural network model

    target_data: list of length (n_cosmo, )
        each element is a list of
        [redshift,
         base-10 log of halo mass,
         base-10 log of halo mass function]

    Returns
    -------
    mse: float
        mean squared error value
    """
    preds_ntup = define_diffsky_hmf_params_namedtuple(preds)

    mse_val = 0.0
    for _loss in target_data:
        z, logmp, loghmf_target = _loss
        loghmf_preds = _predict_cuml_hmf_vmap(
            preds_ntup,
            logmp,
            z,
        )
        mse_val += mse(loghmf_preds, loghmf_target)

    return mse_val


_predict_cuml_hmf_vmap = jjit(vmap(predict_cuml_hmf, in_axes=(0, 0, None)))
