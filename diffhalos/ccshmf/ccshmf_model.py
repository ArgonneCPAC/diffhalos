"""
The predict_ccshmf function gives a differentiable prediction for
the Cumulative Conditional Subhalo Mass Function (CCSHMF),
<Nsub(>mu) | Mhost>, where mu = Msub/Mhost
"""

from jax import jit as jjit

from ..utils import _sig_slope
from .ccshmf_kernels import lg_ccshmf_kern, lg_differential_cshmf_kern

from ..calibrations.ccshmf_cal import DEFAULT_CCSHMF_PARAMS  # noqa

from ..calibrations.ccshmf_cal import (
    YTP_Params,
    YLO_Params,
    CCSHMF_Params,
)

YTP_XTP = 13.0
YLO_XTP = 13.0

__all__ = (
    "predict_ccshmf",
    "predict_differential_cshmf",
)


@jjit
def predict_ccshmf(params, lgmhost, lgmu):
    """
    Model for the cumulative conditional subhalo mass function, CCSHMF,
    defined as <Nsub(>mu) | Mhost>, where mu = Msub/Mhost,
    and both subhalo and host halo masses are peak historical masses

    Parameters
    ----------
    params: namedtuple
        parameters of the fitting function,
        with typical values set by DEFAULT_CCSHMF_PARAMS;
        in detail, params = (ytp_params, ylo_params), where typical values are
        ytp_params = DEFAULT_YTP_PARAMS and ylo_params = DEFAULT_YLO_PARAMS

    lgmhost: float or ndarray of shape (n, )
        base-10 log of host halo mass, in Msun

    lgmu: float or ndarray of shape (n, )
        base-10 log of subhalo-to-host-halo mass

    Returns
    -------
    lg_ccshmf: float or ndarray of shape (n, )
        base-10 log of the CCSHMF, in 1/Mpc^3
    """
    params = CCSHMF_Params(*params)
    ytp = _ytp_model(params.ytp_params, lgmhost)
    ylo = _ylo_model(params.ylo_params, lgmhost)
    lg_ccshmf = lg_ccshmf_kern((ytp, ylo), lgmu)
    return lg_ccshmf


@jjit
def predict_differential_cshmf(params, lgmhost, lgmu):
    """
    Model for the differential conditional subhalo mass function, CCSHMF,
    defined as <Nsub(mu) | Mhost>, where mu = Msub/Mhost,
    and both subhalo and host halo masses are peak historical masses

    Parameters
    ----------
    params: namedtuple
        parameters of the fitting function,
        with typical values set by DEFAULT_CCSHMF_PARAMS;
        in detail, params = (ytp_params, ylo_params), where typical values are
        ytp_params = DEFAULT_YTP_PARAMS and ylo_params = DEFAULT_YLO_PARAMS

    lgmhost: float or ndarray of shape (n, )
        base-10 log of host halo mass, in Msun

    lgmu: float or ndarray of shape (n, )
        base-10 log of subhalo-to-host-halo mass

    Returns
    -------
    lg_diff_ccshmf: float or ndarray of shape (n, )
        base-10 log of the CCSHMF, in 1/Mpc^3
    """
    params = CCSHMF_Params(*params)
    ytp = _ytp_model(params.ytp_params, lgmhost)
    ylo = _ylo_model(params.ylo_params, lgmhost)
    lg_diff_ccshmf = lg_differential_cshmf_kern((ytp, ylo), lgmu)
    return lg_diff_ccshmf


@jjit
def _ytp_model(ytp_params, lgmhost):
    """
    Model for the ``ytp`` sigmoid parameters

    Parameters
    ----------
    ytp_params: tuple
        Ytp model params

    lgmhost: float or ndarray of shape (n, )
        base-10 log of host halo mass, in Msun

    Returns
    -------
    ytp: float or ndarray of shape (n, )
        ytp model parameters
    """
    ytp_params = YTP_Params(*ytp_params)
    ytp = _sig_slope(
        lgmhost,
        YTP_XTP,
        ytp_params.ytp_ytp,
        ytp_params.ytp_x0,
        ytp_params.ytp_k,
        ytp_params.ytp_ylo,
        ytp_params.ytp_yhi,
    )
    return ytp


@jjit
def _ylo_model(ylo_params, lgmhost):
    """
    Model for the ``ylo`` sigmoid parameters

    Parameters
    ----------
    ylo_params: tuple
        Ylo model params

    lgmhost: float or ndarray of shape (n, )
        base-10 log of host halo mass, in Msun

    Returns
    -------
    ylo: float or ndarray of shape (n, )
        ylo model parameters
    """
    ylo_params = YLO_Params(*ylo_params)
    ylo = _sig_slope(
        lgmhost,
        YLO_XTP,
        ylo_params.ylo_ytp,
        ylo_params.ylo_x0,
        ylo_params.ylo_k,
        ylo_params.ylo_ylo,
        ylo_params.ylo_yhi,
    )
    return ylo
