"""
The ``predict_cuml_hmf`` and ``predict_differential_hmf`` functions
give differentiable implementations for the cumulative
and differential mass functions, respectively, for simulated host halos,
and take as input flat parameter arrays. These are both functions of mp,
the peak historical mass of the main progenitor halo.
"""

from collections import OrderedDict, namedtuple

from jax import numpy as jnp
from jax import grad
from jax import jit as jjit
from jax import vmap

from .hmf_kernels import lg_hmf_kern
from ..calibrations.hmf_cal import DEFAULT_HMF_PARAMS, HMF_Params
from ..utils.sigmoid_utils import _sig_slope, _sigmoid

YTP_XTP = 3.0
X0_XTP = 3.0
HI_XTP = 3.0


FLAT_HMF_PDICT = OrderedDict()
FLAT_HMF_PDICT.update(**DEFAULT_HMF_PARAMS.ytp_params._asdict())
FLAT_HMF_PDICT.update(**DEFAULT_HMF_PARAMS.x0_params._asdict())
FLAT_HMF_PDICT.update(**DEFAULT_HMF_PARAMS.lo_params._asdict())
FLAT_HMF_PDICT.update(**DEFAULT_HMF_PARAMS.hi_params._asdict())

FlatHMFParams = namedtuple("FlatHMFParams", FLAT_HMF_PDICT.keys())
DEFAULT_FLAT_HMF_PARAMS = FlatHMFParams(**FLAT_HMF_PDICT)

__all__ = ("predict_cuml_hmf", "predict_diff_hmf")


@jjit
def predict_cuml_hmf(params, logmp, redshift):
    """Predict the cumulative comoving number density of host halos

    Parameters
    ----------
    params: namedtuple
        flat HMF parameters namedtuple

    logmp: array of shape (n_halos, )
        base-10 log of halo mass, in Msun

    redshift: float
        redshift value

    Returns
    -------
    lg_cuml_hmf: array of shape (n_halos, )
        base-10 log of cumulative comoving number density n(>logmp),
        in comoving (1/Mpc)**3

        Note that both number density and halo mass are defined in
        physical units (not h=1 units)

    """
    hmf_params = _get_singlez_cuml_hmf_params(params, redshift)
    return lg_hmf_kern(hmf_params, logmp)


@jjit
def _get_singlez_cuml_hmf_params(params, redshift):
    ytp = _ytp_vs_redshift(params, redshift)
    x0 = _x0_vs_redshift(params, redshift)
    lo = _lo_vs_redshift(params, redshift)
    hi = _hi_vs_redshift(params, redshift)
    hmf_params = HMF_Params(ytp, x0, lo, hi)
    return hmf_params


@jjit
def _ytp_vs_redshift(params, redshift):
    p = (params.ytp_ytp, params.ytp_x0, params.ytp_k, params.ytp_ylo, params.ytp_yhi)
    return _sig_slope(redshift, YTP_XTP, *p)


@jjit
def _x0_vs_redshift(params, redshift):
    p = (params.x0_ytp, params.x0_x0, params.x0_k, params.x0_ylo, params.x0_yhi)
    return _sig_slope(redshift, X0_XTP, *p)


@jjit
def _lo_vs_redshift(params, redshift):
    p = (params.lo_x0, params.lo_k, params.lo_ylo, params.lo_yhi)
    return _sigmoid(redshift, *p)


@jjit
def _hi_vs_redshift(params, redshift):
    p = (params.hi_ytp, params.hi_x0, params.hi_k, params.hi_ylo, params.hi_yhi)
    return _sig_slope(redshift, HI_XTP, *p)


@jjit
def _diff_hmf_grad_kern(params, logmp, redshift):
    lgcuml_nd_pred = predict_cuml_hmf(params, logmp, redshift)
    cuml_nd_pred = 10**lgcuml_nd_pred
    return -cuml_nd_pred


_A = (None, 0, None)
_predict_diff_hmf = jjit(vmap(grad(_diff_hmf_grad_kern, argnums=1), in_axes=_A))


@jjit
def predict_diff_hmf(params, logmp, redshift):
    """Predict the differential comoving number density of host halos

    Parameters
    ----------
    params: namedtuple
        flat HMF parameters namedtuple

    logmp: array of shape (n_halos, )
        base-10 log of halo mass, in Msun

    redshift: float
        redshift value

    Returns
    -------
    hmf: array of shape (n_halos, )
        Differential comoving number density dn(logmp)/dlogmp,
        in comoving (1/Mpc)**3 / dex

        Note that both number density and halo mass are defined in
        physical units (not h=1 units)

    """
    hmf = _predict_diff_hmf(params, logmp, redshift)
    return hmf


@jjit
def wrapper_ndarray_diffsky_diff_hmf(params, logmp, redshift):
    """
    Generates halo mass function model predictions for
    the differential model from the cumulative;
    this is a wrapper on ``diffsky_diff_hmf`` to allow
    to pass the HMF parameters as an array instead of a namedtuple

    Parameters
    ----------
    params: ndarray of shape (n_hmf_params, )
        halo mass function parameters as an array

    logmp: ndarray of shape (n_halos, )
        base-10 log of halo mass, in Msun

    redshift: float
        redshift value

    Returns
    -------
    lg_hmf: ndarray of shape (n_halos, )
        base-10 log of differential comoving halo mass function,
        in comoving (1/Mpc)**3
    """
    params_ntup = FlatHMFParams(*params)
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
    log_cuml_hmf = predict_cuml_hmf(hmf_params, logm, z)
    return 10**log_cuml_hmf


"""Gradient of n(>logmp) wrt logmp"""
_fhm_flat_grad = jjit(
    vmap(
        grad(_cumlhmf_from_logcumlhmf_flat, argnums=(1,)),
        in_axes=(None, 0, None),
    )
)
