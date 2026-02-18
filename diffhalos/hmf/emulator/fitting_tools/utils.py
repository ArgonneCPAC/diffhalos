"""Useful utilities for fitting"""

from jax import numpy as jnp
from jax import jit as jjit

__all__ = ("mse",)


@jjit
def mse(pred, target):
    """
    Mean squared error loss function for optimization

    Parameters
    ----------
    pred: ndarray of shape (n_data,)
        predicted values by the model

    target: ndarray of shape (n_data,)
        target data

    Returns
    -------
    mse_val: float
        mse value
    """
    diff = pred - target
    mse_val = jnp.mean(diff**2)

    return mse_val
