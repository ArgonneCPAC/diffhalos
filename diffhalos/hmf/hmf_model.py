"""
The ``predict_cuml_hmf`` and ``predict_diff_hmf`` functions
give differentiable implementations for the cumulative and differential
mass functions, respectively, for simulated host halos.
These are both functions of mp,
the peak historical mass of the main progenitor halo.
"""

from jax import grad
from jax import jit as jjit
from jax import vmap
from jax import numpy as jnp

from ..cosmology import flat_wcdm, DEFAULT_COSMOLOGY

from .hmf_kernels import lg_hmf_kern
from ..calibrations.hmf_cal import DEFAULT_HMF_PARAMS, HMF_Params  # noqa
from ..cosmology.geometry_utils import (
    spherical_shell_comoving_volume,
    compute_volume_from_sky_area,
)
from ..utils.sigmoid_utils import _sig_slope, _sigmoid
from ..defaults import FULL_SKY_AREA

YTP_XTP = 3.0
X0_XTP = 3.0
HI_XTP = 3.0

N_HMF_GRID = 500

__all__ = (
    "predict_cuml_hmf",
    "predict_diff_hmf",
    "halo_lightcone_weights",
    "get_mean_nhalos_from_volume",
    "get_mean_nhalos_from_sky_area",
)


@jjit
def predict_cuml_hmf(params, logmp, redshift):
    """Predict the cumulative comoving number density of host halos

    Parameters
    ----------
    params: namedtuple
        fitting function parameters

    logmp: ndarray of shape (n_halos, )
        base-10 log of halo mass, in Msun

    redshift: float
        redshift value

    Returns
    -------
    lg_cuml_hmf: ndarray shape (n_halos, )
        base-10 log of cumulative comoving number density n(>logmp),
        in comoving (1/Mpc)**3

        Note that both number density and halo mass are defined in
        physical units (not h=1 units)
    """
    hmf_params = _get_singlez_cuml_hmf_params(params, redshift)
    return lg_hmf_kern(hmf_params, logmp)


@jjit
def _get_singlez_cuml_hmf_params(params, redshift):
    ytp = _ytp_vs_redshift(params.ytp_params, redshift)
    x0 = _x0_vs_redshift(params.x0_params, redshift)
    lo = _lo_vs_redshift(params.lo_params, redshift)
    hi = _hi_vs_redshift(params.hi_params, redshift)
    hmf_params = HMF_Params(ytp, x0, lo, hi)
    return hmf_params


@jjit
def _ytp_vs_redshift(params, redshift):
    p = (
        params.ytp_ytp,
        params.ytp_x0,
        params.ytp_k,
        params.ytp_ylo,
        params.ytp_yhi,
    )
    return _sig_slope(redshift, YTP_XTP, *p)


@jjit
def _x0_vs_redshift(params, redshift):
    p = (
        params.x0_ytp,
        params.x0_x0,
        params.x0_k,
        params.x0_ylo,
        params.x0_yhi,
    )
    return _sig_slope(redshift, X0_XTP, *p)


@jjit
def _lo_vs_redshift(params, redshift):
    p = (
        params.lo_x0,
        params.lo_k,
        params.lo_ylo,
        params.lo_yhi,
    )
    return _sigmoid(redshift, *p)


@jjit
def _hi_vs_redshift(params, redshift):
    p = (
        params.hi_ytp,
        params.hi_x0,
        params.hi_k,
        params.hi_ylo,
        params.hi_yhi,
    )
    return _sig_slope(redshift, HI_XTP, *p)


@jjit
def _diff_hmf_grad_kern(params, logmp, redshift):
    lgcuml_nd_pred = predict_cuml_hmf(params, logmp, redshift)
    cuml_nd_pred = 10**lgcuml_nd_pred
    return -cuml_nd_pred


_A = (None, 0, None)
_predict_diff_hmf = jjit(
    vmap(
        grad(_diff_hmf_grad_kern, argnums=1),
        in_axes=_A,
    )
)


@jjit
def predict_diff_hmf(params, logmp, redshift):
    """Predict the differential comoving number density of host halos

    Parameters
    ----------
    params: namedtuple
        fitting function parameters

    logmp: ndarray of shape (n_halos, )
        base-10 log of halo mass, in Msun

    redshift: float
        redshift value

    Returns
    -------
    hmf: ndarray of shape (n_halos, )
        base-10 log of fifferential comoving number density dn(logmp)/dlogmp,
        in comoving (1/Mpc)**3 / dex

        Note that both number density and halo mass are defined in
        physical units (not h=1 units)
    """
    hmf = jnp.log10(_predict_diff_hmf(params, logmp, redshift))
    return hmf


_dn_dlgm_kern = jjit(grad(_diff_hmf_grad_kern, argnums=1))


@jjit
def _dn_dm_dz_kern(lgm, z, hmf_params, cosmo_params):
    dn_dm_dv = _dn_dlgm_kern(hmf_params, lgm, z)
    dv_dz = flat_wcdm.differential_comoving_volume_at_z(z, *cosmo_params)
    dn_dm_dz = dn_dm_dv * dv_dz
    return dn_dm_dz


_A = (0, 0, None, None)
predict_dn_dlgm_dz = jjit(vmap(_dn_dm_dz_kern, in_axes=_A))


@jjit
def halo_lightcone_weights(
    lgmp,
    redshift,
    sky_area_degsq,
    hmf_params=DEFAULT_HMF_PARAMS,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    """
    Computes lightcone halo weights on a grid
    of redshift and mass from the input

    Parameters
    ----------
    lgmp: ndarray of shape (n_m, )
        base-10 log of halo mass, in Msun

    redshift: ndarray of shape (n_z, )
        redshift

    sky_area_degsq: float
        sky area covered by lightcone, in deg^2

    hmf_params: namedtuple
        halo mass function parameters

    cosmo_params: namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    Returns
    -------
    nhalos: ndarray of shape (n_m*n_z, )
        weighted halo abundance per (mass, redshift) cell
    """
    z_min = redshift.min()
    z_max = redshift.max()
    lgmp_min = lgmp.min()
    lgmp_max = lgmp.max()

    # set up a integration grid in redshift
    z_grid = jnp.linspace(z_min, z_max, N_HMF_GRID)

    # compute the comoving volume of a thin shell at each grid point
    fsky = sky_area_degsq / FULL_SKY_AREA
    vol_shell_grid_mpc = fsky * spherical_shell_comoving_volume(
        z_grid,
        cosmo_params,
    )

    # at each grid point, compute <Nhalos> for the shell volume
    nd_lgmp_min = 10 ** predict_cuml_hmf(hmf_params, lgmp_min, z_grid)
    nd_lgmp_max = 10 ** predict_cuml_hmf(hmf_params, lgmp_max, z_grid)
    nhalos_per_mpc3 = nd_lgmp_min - nd_lgmp_max
    nhalos_per_shell = vol_shell_grid_mpc * nhalos_per_mpc3

    # total number of halos is the sum over shells
    nhalos_tot = nhalos_per_shell.sum()

    # compute relative abundance of halos via weights ~ ∂n/∂z∂m
    _weights = predict_dn_dlgm_dz(lgmp, redshift, hmf_params, cosmo_params)
    weights = _weights / _weights.sum()

    # compute relative number of halos per shell
    nhalos = nhalos_tot * weights

    return nhalos


@jjit
def get_mean_nhalos_from_volume(
    redshift,
    volume_com_mpc,
    hmf_params,
    lgmp_min,
    lgmp_max,
):
    """
    Compute the mean number of halos at a single redshift,
    for masses between two values, given the volume

    Parameters
    ----------
    redshift: ndarray of shape (n_z, )
        redshift value

    volume_com_mpc: float
        comoving volume of the generated population, in Mpc^3

    hmf_params: namedtuple
        halo mass function parameters

    lgmp_min: float
        base-10 log of minimum halo mass, in Msun

    lgmp_max: float
        base-10 log of maximum halo mass, in Msun

    Returns
    -------
    mean_nhalos: ndarray of shape (n_z, )
        mean halo counts
    """
    mean_nhalos_lgmin = _compute_nhalos_tot(
        hmf_params,
        lgmp_min,
        redshift,
        volume_com_mpc,
    )
    mean_nhalos_lgmax = _compute_nhalos_tot(
        hmf_params,
        lgmp_max,
        redshift,
        volume_com_mpc,
    )
    mean_nhalos = mean_nhalos_lgmin - mean_nhalos_lgmax

    return mean_nhalos


@jjit
def get_mean_nhalos_from_sky_area(
    redshift,
    sky_area_degsq,
    cosmo_params,
    hmf_params,
    lgmp_min,
    lgmp_max,
):
    """
    Compute the mean number of halos at a single redshift,
    for masses between two values, given the sky area

    Parameters
    ----------
    redshift: ndarray of shape (n_z, )
        redshift value

    sky_area_degsq: float
        sky area, in deg^2

    cosmo_params: namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    hmf_params: namedtuple
        halo mass function parameters

    lgmp_min: float
        base-10 log of minimum halo mass, in Msun

    lgmp_max: float
        base-10 log of maximum halo mass, in Msun

    Returns
    -------
    mean_nhalos: ndarray of shape (n_z, )
        mean halo counts
    """
    volume_com_mpc = compute_volume_from_sky_area(
        redshift,
        sky_area_degsq,
        cosmo_params,
    )

    mean_nhalos = get_mean_nhalos_from_volume(
        redshift,
        volume_com_mpc,
        hmf_params,
        lgmp_min,
        lgmp_max,
    )

    return mean_nhalos


@jjit
def _compute_nhalos_tot(
    hmf_params,
    lgmp_min,
    redshift,
    volume_com_mpc,
):
    """
    Computes the total number of halos that are expected
    to be found at requested redshift within the input volume,
    and with a provided minimum particle mass

    Parameters
    ----------
    hmf_params: namedtuple
        HMF parameters named tuple

    lgmp_min: float
        base-10 log of the minimum mass

    redshift: float
        redshift value

    volume_com_mpc: float
        comoving volume, in comoving Mpc^3

    Returns
    -------
    nhalos_tot: float
        halo abundance at input redshift and within the input volume,
        considering the minimum mass cut
    """
    nhalos_per_mpc3 = 10 ** predict_cuml_hmf(hmf_params, lgmp_min, redshift)
    nhalos_tot = nhalos_per_mpc3 * volume_com_mpc

    return nhalos_tot
