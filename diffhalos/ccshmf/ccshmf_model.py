"""
The predict_ccshmf function gives a differentiable prediction for
the Cumulative Conditional Subhalo Mass Function (CCSHMF),
<Nsub(>mu) | Mhost>, where mu = Msub/Mhost
"""

from jax import vmap
from jax import jit as jjit
from jax import random as jran

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
    "subhalo_lightcone_weights",
)


@jjit
def _ccshmf_kern(params, lgmhost, lgmu):
    """
    Model for the cumulative conditional subhalo mass function, CCSHMF,
    defined as <Nsub(>mu) | Mhost>, where mu = Msub/Mhost,
    and both subhalo and host halo masses are peak historical masses;
    this works for a single value of Mhost

    Parameters
    ----------
    params: namedtuple
        parameters of the fitting function,
        with typical values set by DEFAULT_CCSHMF_PARAMS;
        in detail, params = (ytp_params, ylo_params), where typical values are
        ytp_params = DEFAULT_YTP_PARAMS and ylo_params = DEFAULT_YLO_PARAMS

    lgmhost:  ndarray of shape (float, )
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


"""
Model for the cumulative conditional subhalo mass function,
for multiple values of Mhost,
by vmapping ``_ccshmf_kern``,
returning a ndarray of shape (n_m, n_mu)
"""
predict_ccshmf = jjit(vmap(_ccshmf_kern, in_axes=(None, 0, None)))


@jjit
def _differential_cshmf(params, lgmhost, lgmu):
    """
    Model for the differential conditional subhalo mass function, CCSHMF,
    defined as <Nsub(mu) | Mhost>, where mu = Msub/Mhost,
    and both subhalo and host halo masses are peak historical masses;
    this works for a single value of Mhost

    Parameters
    ----------
    params: namedtuple
        parameters of the fitting function,
        with typical values set by DEFAULT_CCSHMF_PARAMS;
        in detail, params = (ytp_params, ylo_params), where typical values are
        ytp_params = DEFAULT_YTP_PARAMS and ylo_params = DEFAULT_YLO_PARAMS

    lgmhost: float
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


"""
Model for the differential conditional subhalo mass function, CCSHMF,
for multiple values of Mhost,
by vmapping ``_differential_cshmf``,
returning a ndarray of shape (n_m, n_mu)
"""
predict_differential_cshmf = jjit(
    vmap(_differential_cshmf, in_axes=(None, 0, None)),
)


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


@jjit
def subhalo_lightcone_weights(
    ran_key,
    lgmp,
    lgmu,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
):
    """
    Computes lightcone halo weights on a grid
    of redshift and mass from the input

    Parameters
    ----------
    ran_key: jax.random.PRNGKey
        random key

    lgmp: ndarray of shape (n_m, )
        base-10 log of halo mass, in Msun

    lgmu: ndarray of shape (n_mu, )
        base-10 log of mu=Msub/Mhost

    ccshmf_params: namedtuple
        conditional subhalo halo mass function parameters

    Returns
    -------
    nsubhalos: ndarray of shape (n_m, n_mu)
        weighted subhalo abundance per (mass, mu) point
    """
    lgmu_min = lgmu.min()
    lgmu_max = lgmu.max()

    # at each grid point, compute <Nsubhalos> for the shell volume
    nsub_lgmu_min = 10 ** predict_ccshmf(ccshmf_params, lgmp, lgmu_min)
    nsub_lgmu_max = 10 ** predict_ccshmf(ccshmf_params, lgmp, lgmu_max)
    nsub_per_host = nsub_lgmu_min - nsub_lgmu_max

    # total number of subhalos as the sum over hosts
    uran_key, counts_key = jran.split(ran_key, 2)
    subhalo_counts_per_halo = jran.poisson(counts_key, nsub_per_host)
    nsub_tot = subhalo_counts_per_halo.sum()

    # compute relative abundance of subhalos
    _weights = 10 ** predict_differential_cshmf(ccshmf_params, lgmp, lgmu)
    weights = _weights / _weights.sum()

    # compute relative number of subhalos
    nsubhalos = nsub_tot * weights

    return nsubhalos
