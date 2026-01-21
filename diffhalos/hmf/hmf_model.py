# flake8: noqa: E402
"""
Functions using ``halox`` for HMF computations
See https://halox.readthedocs.io/en/latest/index.html
"""

from jax import config

config.update("jax_enable_x64", True)

from jax import grad
from jax import jit as jjit
from jax import vmap
from jax import numpy as jnp

from dsps.cosmology import flat_wcdm

import halox

from ..utils.geometry_utils import (
    spherical_shell_comoving_volume,
    compute_volume_from_sky_area,
)
from ..defaults import FULL_SKY_AREA
from ..utils.integration import cumtrapz

from ..cosmology import DEFAULT_COSMOLOGY_JAXCOSMO
from ..cosmology.cosmo_conversion import jaxcosmo_to_dsps_cosmology

from ..defaults import DELTA_C

N_HMF_GRID = 500

LG_HMF_CUML_MIN = -16.0

LOGMP_TABLE_NMASS = 500
LOGMP_TABLE_MIN = 6.0
LOGMP_TABLE_MAX = 17.0
LOGMP_TABLE = jnp.linspace(
    LOGMP_TABLE_MIN,
    LOGMP_TABLE_MAX,
    LOGMP_TABLE_NMASS,
)

__all__ = (
    "predict_differential_hmf",
    "predict_cuml_hmf",
    "halo_lightcone_weights",
    "get_mean_nhalos_from_volume",
    "get_mean_nhalos_from_sky_area",
)


@jjit
def predict_differential_hmf(cosmo_params, logmp, redshift, delta_c):
    """
    Predict the differential comoving number density of host halos,
    using the Tinker08 HMF implementation of Halox

    Parameters
    ----------
    cosmo_params: jax-cosmo parameters object
        cosmological paramteters parameters

    logmp: ndarray of shape (n_halos, )
        base-10 log of halo mass, in Msun

    redshift: float
        redshift value

    delta_c: float
        overdensity threshold

    Returns
    -------
    lg_hmf: ndarray of shape (n_halos, )
        base-10 log of fifferential comoving number density dn(logmp)/dlogmp,
        in comoving (1/Mpc)**3 / dex

        Note that both number density and halo mass are defined in
        physical units (not h=1 units)
    """
    # convert mass to units of Msun/h
    logmp += cosmo_params.h

    # compute the differential HMF using Halox
    lg_hmf = halox.hmf.tinker08_mass_function(
        10**logmp,
        redshift,
        cosmo_params,
        delta_c=delta_c,
    )

    # restore to units without h
    lg_hmf *= cosmo_params.h**3

    # convert dlnM to dlogM
    lg_hmf /= jnp.log(10.0)

    # take the log10 of the HMF
    lg_hmf = jnp.log10(lg_hmf)

    return lg_hmf


@jjit
def _cuml_hmf_interp(
    cosmo_params,
    redshift,
    logmp,
    logmp_min,
    logmp_max,
    delta_c,
):

    # predict the differential HMF
    diff_hmf_table = 10 ** predict_differential_hmf(
        cosmo_params,
        LOGMP_TABLE,
        redshift,
        delta_c,
    )

    # compute the cumulative HMF n(<logm)
    cuml_hmf_table = cumtrapz(LOGMP_TABLE, diff_hmf_table)

    # get the cumulative n(>logm)
    cuml_hmf_min = jnp.interp(logmp_min, LOGMP_TABLE, cuml_hmf_table)
    cuml_hmf_max = jnp.interp(logmp_max, LOGMP_TABLE, cuml_hmf_table)
    cuml_hmf_tot = cuml_hmf_min - cuml_hmf_max
    cuml_hmf = cuml_hmf_tot - jnp.interp(logmp, LOGMP_TABLE, cuml_hmf_table)

    return cuml_hmf


@jjit
def predict_cuml_hmf(
    cosmo_params,
    logmp,
    redshift,
    delta_c,
):
    """
    Predict the cumulative comoving number density of host halos

    Parameters
    ----------
    cosmo_params: jax-cosmo parameters object
        cosmological paramteters parameters

    logmp: ndarray of shape (n_halos, )
        base-10 log of halo mass, in Msun;
        note that minimum mass cannot be less than ``LOGMP_TABLE_MIN``
        and maximum mass cannot be greater than ``LOGMP_TABLE_MAX``

    redshift: float
        redshift value

    delta_c: float
        overdensity threshold

    dlogmp_pad: float
        padding to the last mass for the computation
        so that the resulting shape of the cumulative HMF
        is the same as the input mass array, in Msun

    Returns
    -------
    lg_cuml_hmf: ndarray shape (n_halos, )
        base-10 log of cumulative comoving number density n(>logmp),
        in comoving (1/Mpc)**3

        Note that both number density and halo mass are defined in
        physical units (not h=1 units)
    """
    logmp_min = logmp.min()
    logmp_max = logmp.max()

    # compute the cumulative HMF at requested
    # redshift and mass via interpolation using a fixed table
    cuml_hmf = _cuml_hmf_interp(
        cosmo_params,
        redshift,
        logmp,
        logmp_min,
        logmp_max,
        delta_c,
    )

    # take the log10 of the cumulative HMF
    lg_cuml_hmf = jnp.log10(cuml_hmf)

    # guard aginst -inf values at very high masses
    lg_cuml_hmf = jnp.where(
        lg_cuml_hmf > LG_HMF_CUML_MIN,
        lg_cuml_hmf,
        LG_HMF_CUML_MIN,
    )

    return lg_cuml_hmf


"""
Predict the cumulative HMF on a grid of redshifts,
by vmapping ``predict_cuml_hmf`` over redshift
"""
predict_cuml_hmf_multiz = jjit(
    vmap(
        predict_cuml_hmf,
        in_axes=(None, None, 0, None),
    )
)


@jjit
def _diff_hmf_grad_kern(cosmo_params, logmp, redshift, delta_c):
    lgcuml_nd_pred = predict_cuml_hmf(
        cosmo_params,
        logmp,
        redshift,
        delta_c,
    )
    cuml_nd_pred = 10**lgcuml_nd_pred
    return -cuml_nd_pred


_dn_dlgm_kern = jjit(grad(_diff_hmf_grad_kern, argnums=1))


@jjit
def _dn_dm_dz_kern(lgm, z, cosmo_params, delta_c):
    dn_dm_dv = _dn_dlgm_kern(cosmo_params, lgm, z, delta_c)
    dsps_cosmo = jaxcosmo_to_dsps_cosmology(cosmo_params)
    dv_dz = flat_wcdm.differential_comoving_volume_at_z(z, *dsps_cosmo)
    dn_dm_dz = dn_dm_dv * dv_dz
    return dn_dm_dz


_A = (0, 0, None, None)
predict_dn_dlgm_dz = jjit(vmap(_dn_dm_dz_kern, in_axes=_A))


@jjit
def halo_lightcone_weights(
    lgmp,
    redshift,
    sky_area_degsq,
    cosmo_params=DEFAULT_COSMOLOGY_JAXCOSMO,
    delta_c=DELTA_C,
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

    cosmo_params: jax-cosmo parameters object
        cosmological paramteters parameters

    delta_c: float
        overdensity threshold

    dlogmp_pad: float
        padding to the last mass for the computation
        so that the resulting shape of the cumulative HMF
        is the same as the input mass array, in Msun

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
    nd_lgmp_min = 10 ** predict_cuml_hmf_multiz(
        cosmo_params,
        lgmp_min,
        z_grid,
        delta_c,
    )
    nd_lgmp_max = 10 ** predict_cuml_hmf_multiz(
        cosmo_params,
        lgmp_max,
        z_grid,
        delta_c,
    )
    nhalos_per_mpc3 = nd_lgmp_min - nd_lgmp_max
    nhalos_per_shell = vol_shell_grid_mpc * nhalos_per_mpc3

    # total number of halos is the sum over shells
    nhalos_tot = nhalos_per_shell.sum()

    # compute relative abundance of halos via weights ~ ∂n/∂z∂m
    _weights = predict_dn_dlgm_dz(
        lgmp,
        redshift,
        cosmo_params,
        delta_c,
    )
    weights = _weights / _weights.sum()

    # compute relative number of halos per shell
    nhalos = nhalos_tot * weights

    return nhalos


@jjit
def get_mean_nhalos_from_volume(
    redshift,
    volume_com_mpc,
    cosmo_params,
    lgmp_min,
    lgmp_max,
    delta_c=DELTA_C,
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

    cosmo_params: jax-cosmo parameters object
        cosmological paramteters parameters

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
        cosmo_params,
        lgmp_min,
        redshift,
        volume_com_mpc,
        delta_c=delta_c,
    )
    mean_nhalos_lgmax = _compute_nhalos_tot(
        cosmo_params,
        lgmp_max,
        redshift,
        volume_com_mpc,
        delta_c=delta_c,
    )
    mean_nhalos = mean_nhalos_lgmin - mean_nhalos_lgmax

    return mean_nhalos


@jjit
def get_mean_nhalos_from_sky_area(
    redshift,
    sky_area_degsq,
    cosmo_params,
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

    cosmo_params: jax-cosmo parameters object
        cosmological paramteters parameters

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
        cosmo_params,
        lgmp_min,
        lgmp_max,
    )

    return mean_nhalos


@jjit
def _compute_nhalos_tot(
    cosmo_params,
    lgmp_min,
    redshift,
    volume_com_mpc,
    delta_c=DELTA_C,
):
    """
    Computes the total number of halos that are expected
    to be found at requested redshift within the input volume,
    and with a provided minimum particle mass

    Parameters
    ----------
    cosmo_params: jax-cosmo parameters object
        cosmological paramteters parameters

    lgmp_min: float
        base-10 log of the minimum mass

    redshift: float
        redshift value

    volume_com_mpc: float
        comoving volume, in comoving Mpc^3

    delta_c: float
        overdensity threshold

    dlogmp_pad: float
        padding to the last mass for the computation
        so that the resulting shape of the cumulative HMF
        is the same as the input mass array, in Msun

    Returns
    -------
    nhalos_tot: float
        halo abundance at input redshift and within the input volume,
        considering the minimum mass cut
    """
    nhalos_per_mpc3 = 10 ** predict_cuml_hmf(
        cosmo_params,
        lgmp_min,
        redshift,
        delta_c,
    )
    nhalos_tot = nhalos_per_mpc3 * volume_com_mpc

    return nhalos_tot
