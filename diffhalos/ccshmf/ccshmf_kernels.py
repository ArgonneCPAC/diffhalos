"""
The lg_cuml_cshmf_kern function gives a differentiable prediction for
the Cumulative Conditional Subhalo Mass Function (CCSHMF),
<Nsub(>μ) | Mhost>, where μ = Msub/Mhost.
The lg_cuml_cshmf_kern function is a single-halo kernel
called by ccshmf.predict_cuml_cshmf
"""

from collections import OrderedDict, namedtuple

from jax import grad
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _sig_slope

XTP = -1.0
K = 10
X0 = -0.35
YHI = -3.1

DEFAULT_CCSHMF_KERN_PDICT = OrderedDict(ytp=-0.3, ylo=-0.95)
CCSHMF_Params = namedtuple("CCSHMF_Params", DEFAULT_CCSHMF_KERN_PDICT.keys())
DEFAULT_CCSHMF_KERN_PARAMS = CCSHMF_Params(**DEFAULT_CCSHMF_KERN_PDICT)

__all__ = (
    "lg_ccshmf_kern",
    "ccshmf_kern",
    "lg_diff_cshmf_kern",
)


@jjit
def lg_cuml_cshmf_kern(params, lgmu):
    """
    Computes the base-10 log of the CCSHMF

    Parameters
    ----------
    params: namedtuple
        parameters of the fitting function,
        with typical values set by DEFAULT_CCSHMF_PARAMS;
        in detail, params = (ytp_params, ylo_params), where typical values are
        ytp_params = DEFAULT_YTP_PARAMS and ylo_params = DEFAULT_YLO_PARAMS

    lgmu: ndarray of shape (n, )
        base-10 log of subhalo-to-host-halo mass

    Returns
    -------
    lg_cuml_cshmf: ndarray of shape (n, )
        base-10 log of the cumulative cshmf
    """
    params = CCSHMF_Params(*params)
    lg_cuml_cshmf = _sig_slope(lgmu, XTP, params.ytp, X0, K, params.ylo, YHI)

    return lg_cuml_cshmf


@jjit
def cuml_cshmf_kern(params, lgmu):
    """
    Computes the CCSHMF

    Parameters
    ----------
    params: namedtuple
        parameters of the fitting function,
        with typical values set by DEFAULT_CCSHMF_PARAMS;
        in detail, params = (ytp_params, ylo_params), where typical values are
        ytp_params = DEFAULT_YTP_PARAMS and ylo_params = DEFAULT_YLO_PARAMS

    lgmu: flondarray of shape (n, )at
        base-10 log of subhalo-to-host-halo mass

    Returns
    -------
    ccshmf: ndarray of shape (n, )
        value of the ccshmf
    """
    lg_cuml = lg_cuml_cshmf_kern(params, lgmu)
    ccshmf = 10**lg_cuml

    return ccshmf


"""
Helper function to compute the differential CCSHMF model prediction,
vmapped over ``mu`` axis
"""
_diff_cshmf_kern = jjit(
    vmap(grad(cuml_cshmf_kern, argnums=1), in_axes=(None, 0)),
)


@jjit
def lg_diff_cshmf_kern(params, lgmu):
    """
    Computes the base-10 lof of the differential CCSHMF

    Parameters
    ----------
    params: namedtuple
        parameters of the fitting function,
        with typical values set by DEFAULT_CCSHMF_PARAMS;
        in detail, params = (ytp_params, ylo_params), where typical values are
        ytp_params = DEFAULT_YTP_PARAMS and ylo_params = DEFAULT_YLO_PARAMS

    lgmu: ndarray of shape (n, )
        base-10 log of subhalo-to-host-halo mass

    Returns
    -------
    ccshmf: ndarray of shape (n, )
        value of the ccshmf
    """
    return jnp.log10(-_diff_cshmf_kern(params, lgmu))
