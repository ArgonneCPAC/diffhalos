"""
The predict_ccshmf function gives a differentiable prediction for
the Cumulative Conditional Subhalo Mass Function (CCSHMF),
<Nsub(>mu) | Mhost>, where mu = Msub/Mhost
"""

from functools import partial

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..calibrations.ccshmf_cal import (
    DEFAULT_CCSHMF_PARAMS,
    CCSHMF_Params,
    YLO_Params,
    YTP_Params,
)
from ..utils import _sig_slope
from .ccshmf_kernels import lg_ccshmf_kern, lg_differential_cshmf_kern

YTP_XTP = 13.0
YLO_XTP = 13.0

__all__ = (
    "predict_ccshmf",
    "predict_ccshmf_halopop",
    "predict_differential_cshmf",
    "predict_differential_cshmf_halopop",
    "subhalo_lightcone_weights",
    "compute_mean_subhalo_counts",
)


@jjit
def predict_ccshmf(params, lgmhost, lgmu):
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

    lgmhost:  float
        base-10 log of host halo mass, in Msun

    lgmu: ndarray of shape (n, )
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
by vmapping ``predict_ccshmf``,
returning a ndarray of shape (n_m, n_mu)
"""
predict_ccshmf_halopop = jjit(vmap(predict_ccshmf, in_axes=(None, 0, None)))


@jjit
def predict_differential_cshmf(params, lgmhost, lgmu):
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
by vmapping ``predict_differential_cshmf``,
returning a ndarray of shape (n_m, n_mu)
"""
predict_differential_cshmf_halopop = jjit(
    vmap(predict_differential_cshmf, in_axes=(None, 0, None)),
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
def get_lgmu_cutoff(lgmhost, lgmp_sim, nptcl_cut):
    """
    Get the cutoff mu value for a simulation
    """
    lgmp_cutoff = lgmp_sim + jnp.log10(nptcl_cut)
    lgmu_cutoff = lgmp_cutoff - lgmhost
    return lgmu_cutoff


@jjit
def compute_mean_subhalo_counts(
    lgmhost,
    lgmp_min,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
):
    """
    Computes the mean counts of subhalos,
    given the mass of the host halos

    Parameters
    ----------
    lgmhost: ndarray of shape (n_host, )
        base-10 log of the host halo masses, in Msun

    lgmp_min: float
        base-10 log of the minimum mass, in Msun

    ccshmf_params: namedtuple
        cumulative conditional subhalo mass function parameters

    Returns
    -------
    mean_counts: ndarray of shape (n_host, )
        mean subhalo counts per host halo
    """
    lgmu_cutoff = get_lgmu_cutoff(lgmhost, lgmp_min, 1)
    mean_counts = 10 ** predict_ccshmf(ccshmf_params, lgmhost, lgmu_cutoff)

    return mean_counts


@partial(jjit, static_argnames=["n_logmu"])
def subhalo_lightcone_weights_kern(
    lgmhost,
    lgmp_min,
    n_logmu,
    ccshmf_params,
):
    """
    Computes the weighted subhalo counts that reside
    in a single host halo

    Parameters
    ----------
    lgmhost: ndarray of shape (n_host, )
        base-10 log of the host halo masses, in Msun

    lgmp_min: float
        base-10 log of the minimum mass, in Msun

    ccshmf_params: namedtuple
        cumulative conditional subhalo mass function parameters

    n_logmu: int
        number of mu=Msub/Mhost values to use

    Returns
    -------
    nsubhalos_in_host: ndarray of shape (n_mu, )
        subhalo counts each mu value
    """
    # compute minumum allowed mu value
    lgmu_cutoff = get_lgmu_cutoff(lgmhost, lgmp_min, 1)

    # calcumate array of mu values, given lgmp_min
    lgmu = jnp.linspace(lgmu_cutoff, 0, n_logmu)

    # compute <Nsubhalos> for a single host halo
    subhalo_counts_per_halo = 10 ** predict_ccshmf(
        ccshmf_params,
        lgmhost,
        lgmu_cutoff,
    )

    # compute relative abundance of subhalos
    _weights = 10 ** predict_differential_cshmf(ccshmf_params, lgmhost, lgmu)
    weights = _weights / _weights.sum()

    # compute relative number of subhalos
    nsubhalos_in_host = subhalo_counts_per_halo * weights

    return nsubhalos_in_host, lgmu


""""
Computes the weighted subhalo counts that reside in multiple host halos,
by vmapping ``subhalo_lightcone_weights_kern``,
with output being a ndarray of shape (n_host, n_mu)
"""
subhalo_lightcone_weights = jjit(
    vmap(subhalo_lightcone_weights_kern, in_axes=(0, None, None, None)),
    static_argnames=["n_logmu"],
)
