"""
The ``predict_cuml_hmf`` and ``predict_differential_hmf`` functions
give differentiable implementations for the cumulative and differential
mass functions, respectively, for simulated host halos.
These are both functions of mp,
the peak historical mass of the main progenitor halo.
"""

from jax import grad
from jax import jit as jjit
from jax import vmap
from jax import numpy as jnp

from dsps.cosmology import flat_wcdm
from dsps.cosmology import DEFAULT_COSMOLOGY

from ..lightcone.utils import spherical_shell_comoving_volume
from ..calibrations.hmf_cal import DEFAULT_HMF_PARAMS, HMF_Params  # noqa
from .hmf_kernels import lg_hmf_kern
from ..utils.sigmoid_utils import _sig_slope, _sigmoid
from ..defaults import FULL_SKY_AREA

YTP_XTP = 3.0
X0_XTP = 3.0
HI_XTP = 3.0

N_HMF_GRID = 500

__all__ = (
    "predict_cuml_hmf",
    "predict_differential_hmf",
    "halo_lightcone_weights",
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
_predict_differential_hmf = jjit(
    vmap(
        grad(_diff_hmf_grad_kern, argnums=1),
        in_axes=_A,
    )
)


@jjit
def predict_differential_hmf(params, logmp, redshift):
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
    hmf = jnp.log10(_predict_differential_hmf(params, logmp, redshift))
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
