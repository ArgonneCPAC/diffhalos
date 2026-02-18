"""Model for halo mass function in diffsky"""

import numpy as np

from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap, grad
from jax import random as jran

from diffsky.mass_functions import hmf_model, flat_hmf_model

from ..param_utils.hmf_params import HMF_Params

__all__ = (
    "diffsky_cuml_hmf",
    "diffsky_diff_hmf",
    "diffsky_cuml_hmf_flat",
    "diffsky_diff_hmf_flat",
    "get_binned_diff_from_cuml_hmf",
)


"""Nested HMF model functions"""


@jjit
def diffsky_cuml_hmf(params, logmp, redshift):
    """
    Covenient function that calls the
    halo mass function model prediction from diffsky,
    for model that accepts a nested parameter named tuple

    Parameters
    ----------
    params: namedtuple
        halo mass function parameters

    logmp: ndarray of shape (n_halos,)
        Base-10 log of halo mass, in Msun

    redshift: float
        redshift value

    Returns
    -------
    lg_cuml_hmf: ndarray of shape (n_halos,)
        base-10 log of cumulative comoving number density n(>logmp)
        in units of comoving (1/Mpc)**3
    """
    lg_cuml_hmf = hmf_model.predict_cuml_hmf(params, logmp, redshift)

    return lg_cuml_hmf


@jjit
def diffsky_diff_hmf(params, logmp, redshift):
    """
    Generates halo mass function model predictions for
    the differential model from the cumulative,
    for model that accepts a nested parameter named tuple

    Parameters
    ----------
    params: namedtuple
        halo mass function parameters

    logmp: ndarray of shape (n_halos,)
        Base-10 log of halo mass, in Msun

    redshift: float
        redshift value

    Returns
    -------
    lg_hmf: ndarray of shape (n_halos,)
        base-10 log of differential comoving halo mass function,
        in units of comoving (1/Mpc)**3
    """
    diff_hmf = _diffsky_diff_from_cuml_hmf_jax_grad(params, logmp, redshift)
    lg_hmf = jnp.log10(diff_hmf)

    return lg_hmf


@jjit
def _diffsky_diff_from_cuml_hmf_jax_grad(params, logmp, redshift):
    """Helper function to properly get the gradient of n(>lohmp) wrt logmp"""
    diff_hmf = -_fhm_grad(params, logmp, redshift)[0]
    return diff_hmf


@jjit
def _cumlhmf_from_logcumlhmf(hmf_params, logm, z):
    """Helper function to get n(>logmp) from base-10 log of n(>logmp)"""
    log_cuml_hmf = hmf_model.predict_cuml_hmf(hmf_params, logm, z)
    return 10**log_cuml_hmf


"""Gradient of n(>logmp) wrt logmp"""
_fhm_grad = jjit(
    vmap(
        grad(_cumlhmf_from_logcumlhmf, argnums=(1,)),
        in_axes=(None, 0, None),
    )
)


def get_binned_diff_from_cuml_hmf(
    params,
    logmp,
    redshift,
    n_halo=800_000,
    n_bins=50,
    volume=1.0,
    n_tot=100,
):
    """
    Generates halo mass function model predictions for
    the differential model from the cumulative,
    by sampling from the cumulative and binning using,
    i.e. it uses finite differences instead of ``jax.grad``

    Parameters
    ----------
    params: namedtuple
        halo mass function parameters

    logmp: ndarray of shape (n_halos,)
        Base-10 log of halo mass, in Msun

    redshift: float
        redshift value

    n_halo: int
        number of halos to generate from sampling

    n_bins: int
        number of bins for histogram

    volume: float
        comoving volume, in Mpc^3

    ntot: int
        total number of halos for normalization

    Returns
    -------
    dnhalo_bins: ndarray of shape (n_halo,)
        base-10 log of cumulative comoving halo mass function,
        in units of comoving (1/Mpc)**3

    dlogM_bins: ndarray of shape (n_halo,)
        base-10 log of halo mass, in Msun
    """
    # construct halo mass function CDF
    log_cuml_density = hmf_model.predict_cuml_hmf(params, logmp, redshift)
    cuml_hmf = 10**log_cuml_density
    cuml_hmf = cuml_hmf - cuml_hmf[0]
    cuml_hmf = cuml_hmf / cuml_hmf[-1]

    # generate random key
    ran_key = jran.key(0)

    # sample from uniform random distribution
    uran = jran.uniform(ran_key, shape=n_halo)
    logmhalo_diff_hmf = jnp.interp(uran, cuml_hmf, logmp)

    hist_data = np.histogram(logmhalo_diff_hmf, bins=n_bins, density=False)
    dlogM_bin_edges = hist_data[1]
    dlogM_bins = 0.5 * (dlogM_bin_edges[1:] + dlogM_bin_edges[:-1])
    dnhalo_bins = hist_data[0] / np.diff(dlogM_bin_edges) / n_halo / volume * n_tot

    return dnhalo_bins, dlogM_bins


"""Flat HMF model functions"""


@jjit
def diffsky_cuml_hmf_flat(params, logmp, redshift):
    """
    Covenient function that calls the
    halo mass function model prediction from diffsky,
    for model that accepts a flat parameter named tuple

    Parameters
    ----------
    params: namedtuple
        halo mass function parameters

    logmp: ndarray of shape (n_halos,)
        Base-10 log of halo mass, in Msun

    redshift: float
        redshift value

    Returns
    -------
    lg_cuml_hmf: ndarray of shape (n_halos,)
        base-10 log of cumulative comoving number density n(>logmp)
        in units of comoving (1/Mpc)**3
    """
    lg_cuml_hmf = flat_hmf_model.predict_cuml_hmf(params, logmp, redshift)

    return lg_cuml_hmf


@jjit
def diffsky_diff_hmf_flat(params, logmp, redshift):
    """
    Generates halo mass function model predictions for
    the differential model from the cumulative,
    for model that accepts a flat parameter named tuple

    Parameters
    ----------
    params: namedtuple
        halo mass function parameters

    logmp: ndarray of shape (n_halos,)
        Base-10 log of halo mass, in Msun

    redshift: float
        redshift value

    Returns
    -------
    lg_hmf: ndarray of shape (n_halos,)
        base-10 log of differential comoving halo mass function,
        in units of comoving (1/Mpc)**3
    """
    diff_hmf = _diffsky_diff_from_cuml_hmf_flat_jax_grad(
        params,
        logmp,
        redshift,
    )
    lg_hmf = jnp.log10(diff_hmf)

    return lg_hmf


@jjit
def wrapper_ndarray_diffsky_diff_hmf(params, logmp, redshift):
    """
    Generates halo mass function model predictions for
    the differential model from the cumulative;
    this is a wrapper on ``diffsky_diff_hmf`` to allow
    to pass the HMF parameters as an array instead of a namedtuple

    Parameters
    ----------
    params: ndarray of shape (n_hmf_params,)
        halo mass function parameters

    logmp: ndarray of shape (n_halos,)
        Base-10 log of halo mass, in Msun

    redshift: float
        redshift value

    Returns
    -------
    lg_hmf: ndarray of shape (n_halos,)
        base-10 log of differential comoving halo mass function,
        in units of comoving (1/Mpc)**3
    """
    # params_ntup = diffsky_params_obj.array_to_ntup(params)
    params_ntup = HMF_Params(*params)
    diff_hmf = _diffsky_diff_from_cuml_hmf_flat_jax_grad(
        params_ntup,
        logmp,
        redshift,
    )
    lg_hmf = jnp.log10(diff_hmf)

    return lg_hmf


@jjit
def _diffsky_diff_from_cuml_hmf_flat_jax_grad(params, logmp, redshift):
    """Helper function to properly get the gradient of n(>lohmp) wrt logmp"""
    diff_hmf = -_fhm_flat_grad(params, logmp, redshift)[0]
    return diff_hmf


@jjit
def _cumlhmf_from_logcumlhmf_flat(hmf_params, logm, z):
    """Helper function to get n(>logmp) from base-10 log of n(>logmp)"""
    log_cuml_hmf = flat_hmf_model.predict_cuml_hmf(hmf_params, logm, z)
    return 10**log_cuml_hmf


"""Gradient of n(>logmp) wrt logmp"""
_fhm_flat_grad = jjit(
    vmap(
        grad(_cumlhmf_from_logcumlhmf_flat, argnums=(1,)),
        in_axes=(None, 0, None),
    )
)
