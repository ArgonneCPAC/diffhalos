"""Useful utilities for fitting"""

from functools import partial

from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap

from ...hmf_model import predict_diff_hmf, predict_cuml_hmf
from ...hmf_param_utils import define_diffsky_hmf_params_namedtuple

__all__ = ("mse",)


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
        predicted values by the model

    target: ndarray of shape (n_cosmo, n_hmf_params)
        target data

    Returns
    -------
    mse_val: float
        mse value
    """
    diff = (pred - target) ** 2
    mse_val = jnp.sum(jnp.mean(diff, axis=1))

    return mse_val


# @partial(jjit, static_argnames=["n_cosmo"])
# def mse_loss_diff_hmf_curve(preds, target, loss_data, n_cosmo):
#     """
#     Mean squared error loss function
#     on the halo mass funcion curve

#     Parameters
#     ----------
#     preds: ndarray of shape (n_hmf_params, )
#         hmf parameters predicted by the mlp

#     target: ndarray of shape (n_redshift, n_halo)
#         target hmf values

#     loss_data: list
#         each element is a list of
#         [redshift,
#          base-10 log of halo mass,
#          base-10 log of halo mass function]

#     n_cosmo: int
#         number of sampled cosmologies

#     Returns
#     -------
#     mse: float
#         mean squared error value between hmf from
#         mlp prediction and target data
#     """
#     mse_val = 0.0
#     for i in range(n_cosmo):
#         redshifts, logmp = loss_data[i]
#         preds_loghmf = _predict_diff_hmf_multiz(
#             define_diffsky_hmf_params_namedtuple(preds), logmp, redshifts
#         )

#         diff = (preds_loghmf, target[i]) ** 2
#         mse_val += jnp.sum(jnp.mean(diff, axis=1))

#     return mse_val


# _predict_diff_hmf_multiz = jjit(vmap(predict_diff_hmf, in_axes=(None, 0, 0)))
