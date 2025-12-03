# flake8: noqa: E402
"""Functions to generate Monte Carlo realizations of host halos in a lightcone"""

from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import warnings
from scipy.stats import qmc

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from diffmah.diffmah_kernels import _log_mah_kern
from diffmah.diffmahpop_kernels.bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS
from diffmah.diffmahpop_kernels.mc_bimod_cens import mc_cenpop

from dsps.cosmology import flat_wcdm
from dsps.cosmology import DEFAULT_COSMOLOGY

from ..hmf.hmf_model import halo_lightcone_weights
from ..hmf import hmf_model, mc_hosts
from ..diffmahpop.diffmahnet_utils import mc_mah_cenpop
from .utils import spherical_shell_comoving_volume

N_HMF_GRID = 2_000
DEFAULT_LOGMP_CUTOFF = 10.0
DEFAULT_LOGMP_HIMASS_CUTOFF = 14.5

FULL_SKY_AREA = (4.0 * jnp.pi) * (180.0 / jnp.pi) ** 2

_AXES = (0, None, None, 0, None)
mc_logmp_vmap = jjit(vmap(mc_hosts._mc_host_halos_singlez_kern, in_axes=_AXES))

__all__ = (
    "mc_lightcone_host_halo_mass_function",
    "get_nhalo_weighted_lc_grid",
    "mc_lightcone_host_halo_diffmah",
    "get_weighted_lightcone_grid_host_halo_diffmah",
    "mc_weighted_halo_lightcone",
)


def mc_lightcone_host_halo_mass_function(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    cosmo_params=DEFAULT_COSMOLOGY,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    n_hmf_grid=N_HMF_GRID,
    lgmp_max=mc_hosts.LGMH_MAX,
    nhalos_tot=None,
):
    """
    Generate a Monte Carlo realization of a lightcone of
    host halo mass and redshift, using a linearly spaced grid in redshift

    Parameters
    ----------
    ran_key: jran.key
        random key

    lgmp_min: float
        minimum halo mass, in Msun

    z_min: float
        minimum redshift

    z_max: float
        maximum redshift

    sky_area_degsq: float
        sky area, in deg^2

    cosmo_params: namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    hmf_params: namedtuple
        halo mass function parameters

    n_hmf_grid: int
        number of grid points in redshift
        at which to compute the HMF

    lgmp_max: float
        base-10 log of maximum halo mass, in Msun

    Returns
    -------
    z_halopop: ndarray of shape (n_halos, )
        redshifts distributed randomly within the lightcone volume

    logmp_halopop: ndarray of shape (n_halos, )
        halo masses derived by Monte Carlo sampling the halo mass function
        at the appropriate redshift for each point
    """

    # three randoms: one for Nhalos, one for halo mass, one for redshift
    halo_counts_key, m_key, z_key = jran.split(ran_key, 3)

    # set up a uniform grid in redshift
    z_grid = jnp.linspace(z_min, z_max, n_hmf_grid)

    # compute the comoving volume of a thin shell at each grid point
    fsky = sky_area_degsq / FULL_SKY_AREA
    vol_shell_grid_mpc = fsky * spherical_shell_comoving_volume(z_grid, cosmo_params)

    # at each grid point, compute <Nhalos> for the shell volume
    mean_nhalos_grid = mc_hosts._compute_nhalos_tot(
        hmf_params, lgmp_min, z_grid, vol_shell_grid_mpc
    )
    mean_nhalos_lgmp_max = mc_hosts._compute_nhalos_tot(
        hmf_params, lgmp_max, z_grid, vol_shell_grid_mpc
    )
    mean_nhalos_grid = mean_nhalos_grid - mean_nhalos_lgmp_max

    if nhalos_tot is None:
        # at each grid point, compute a Poisson realization of <Nhalos>
        nhalos_grid = jran.poisson(halo_counts_key, mean_nhalos_grid)
        nhalos_tot = nhalos_grid.sum()

    # compute the CDF of the volume
    weights_grid = mean_nhalos_grid / mean_nhalos_grid.sum()
    cdf_grid = jnp.cumsum(weights_grid)

    # assign redshift via inverse transformation sampling of the halo counts CDF
    uran_z = jran.uniform(z_key, minval=0, maxval=1, shape=(nhalos_tot,))
    z_halopop = jnp.interp(uran_z, cdf_grid, z_grid)

    # randoms used in inverse transformation sampling halo mass
    uran_m = jran.uniform(m_key, minval=0, maxval=1, shape=(nhalos_tot,))

    # draw a halo mass from the HMF at the particular redshift of each halo
    logmp_halopop = mc_logmp_vmap(uran_m, hmf_params, lgmp_min, z_halopop, lgmp_max)

    return z_halopop, logmp_halopop


@jjit
def pdf_weighted_lgmp_grid_singlez(hmf_params, lgmp_grid, redshift):
    """
    Weights for halos at a single redshift

    Parameters
    ----------
    hmf_params: namedtuple
        halo mass function parameters

    lgmp_grid: ndarray of shape (n_m, )
        base-10 log of halo masses, in Msun

    redshift: float
        redshift at which to compute the HMF

    Returns
    -------
    weights_grid: ndarray of shape (n_m, )
        weights PDF value for each halo mass
    """
    weights_grid = hmf_model.predict_differential_hmf(hmf_params, lgmp_grid, redshift)
    weights_grid = weights_grid / weights_grid.sum()

    return weights_grid


"""
Weights for halos at a multiple redshifts,
by vmapping ``pdf_weighted_lgmp_grid_singlez``
"""
_A = (None, None, 0)
pdf_weighted_lgmp_grid_vmap = jjit(vmap(pdf_weighted_lgmp_grid_singlez, in_axes=_A))


@jjit
def get_nhalo_weighted_lc_grid(
    lgmp_grid,
    z_grid,
    sky_area_degsq,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    """
    Compute the number of halos on the input grid of halo mass and redshift

    Parameters
    ----------
    lgmp_grid: ndarray of shape (n_m, )
        base-10 log halo mass, in Msun

    z_grid: ndarray of shape (n_z, )
        redshift values on the grid

    sky_area_degsq: float
        sky area, in deg^2

    cosmo_params: namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    Returns
    -------
    nhalo_weighted_lc_grid: ndarray of shape (n_z, n_m)
        weighted halo counts on a grid of redshift and mass
    """
    # compute the comoving volume of a thin shell at each grid point
    fsky = sky_area_degsq / FULL_SKY_AREA
    vol_shell_grid_mpc = fsky * spherical_shell_comoving_volume(z_grid, cosmo_params)

    # at each grid point, compute <Nhalos> for the shell volume
    mean_nhalos_grid = mc_hosts._compute_nhalos_tot(
        hmf_params, lgmp_grid[0], z_grid, vol_shell_grid_mpc
    )
    mean_nhalos_lgmp_max = mc_hosts._compute_nhalos_tot(
        hmf_params, lgmp_grid[-1], z_grid, vol_shell_grid_mpc
    )
    mean_nhalos_grid = mean_nhalos_grid - mean_nhalos_lgmp_max

    lgmp_weights = pdf_weighted_lgmp_grid_vmap(hmf_params, lgmp_grid, z_grid)

    n_z = z_grid.size
    nhalo_weighted_lc_grid = mean_nhalos_grid.reshape((n_z, 1)) * lgmp_weights

    return nhalo_weighted_lc_grid


# NOTE: need to fix issues with using diffmahnet
def mc_lightcone_host_halo_diffmah(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    cosmo_params=DEFAULT_COSMOLOGY,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    n_hmf_grid=N_HMF_GRID,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
    lgmp_max=mc_hosts.LGMH_MAX,
):
    """
    Generate halo MAHs for host halos sampled from a lightcone

    Parameters
    ----------
    ran_key: jran.key
        random key

    lgmp_min: float
        minimum halo mass, in Msun

    z_min: float
        minimum redshift value

    z_max: float
        maximum redshift value

    sky_area_degsq: float
        sky area, in deg^2

    cosmo_params: namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    logmp_cutoff: float
        base-10 log of minimum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun;
        for logmp < logmp_cutoff, P(θ_MAH | logmp) = P(θ_MAH | logmp_cutoff)

    logmp_cutoff_himass: float
        base-10 log of maximum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun

    lgmp_max: float
        base-10 log of maximum host halo mass, in Msun

    Returns
    -------
    cenpop: dict with keys:
        z_obs: ndarray of shape (n_halos, )
            lightcone redshift

        logmp_obs: ndarray of shape (n_halos, )
            halo mass at the lightcone redshift, in Msun

        mah_params: namedtuple of ndarray's with shape (n_halos, )
            diffmah parameters

        logmp0: narray, shape (n_halos, )
            base-10 log of halo mass at z=0, in Msun
    """

    lc_hmf_key, mah_key = jran.split(ran_key, 2)
    z_obs, logmp_obs_mf = mc_lightcone_host_halo_mass_function(
        lc_hmf_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
        cosmo_params=cosmo_params,
        hmf_params=hmf_params,
        n_hmf_grid=n_hmf_grid,
        lgmp_max=lgmp_max,
    )
    t_obs = flat_wcdm.age_at_z(z_obs, *cosmo_params)
    t_0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t_0)

    logmp_obs_mf_clipped = np.clip(logmp_obs_mf, logmp_cutoff, logmp_cutoff_himass)

    tarr = np.array((10**lgt0,))
    args = (diffmahpop_params, tarr, logmp_obs_mf_clipped, t_obs, mah_key, lgt0)
    mah_params_uncorrected = mc_cenpop(*args)[0]  # mah_params, dmhdt, log_mah

    logmp_obs_orig = _log_mah_kern(mah_params_uncorrected, t_obs, lgt0)
    delta_logmh_clip = logmp_obs_orig - logmp_obs_mf
    mah_params = mah_params_uncorrected._replace(
        logm0=mah_params_uncorrected.logm0 - delta_logmh_clip
    )

    logmp0 = _log_mah_kern(mah_params, 10**lgt0, lgt0)
    logmp_obs = _log_mah_kern(mah_params, t_obs, lgt0)

    fields = ("z_obs", "t_obs", "logmp_obs", "mah_params", "logmp0")
    values = (z_obs, t_obs, logmp_obs, mah_params, logmp0)
    cenpop_out = dict()
    for key, value in zip(fields, values):
        cenpop_out[key] = value

    return cenpop_out


# NOTE: need to fix issues with using diffmahnet
def get_weighted_lightcone_grid_host_halo_diffmah(
    ran_key,
    lgmp_grid,
    z_grid,
    sky_area_degsq,
    cosmo_params=DEFAULT_COSMOLOGY,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
):
    """
    Compute the number of halos on the input grid of halo mass and redshift

    Parameters
    ----------
    ran_key: jran.key
        random key

    lgmp_grid: ndarray of shape (n_m, )
        grid of base-10 log of halo mass, in Msun

    z_grid: ndarray of shape (n_z, )
        grid of redshift

    sky_area_degsq: float
        sky area, in deg^2

    cosmo_params: namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    hmf_params: namedtuple
        halo mass function parameters

    diffmahpop_params: namedtuple
        diffmahpop parameters

    logmp_cutoff: float
        base-10 log of minimum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun;
        for logmp < logmp_cutoff, P(θ_MAH | logmp) = P(θ_MAH | logmp_cutoff)

    logmp_cutoff_himass: float
        base-10 log of maximum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun

    Returns
    -------
    cenpop: dict with keys:
        z_obs: ndarray of shape (n_z*n_m, )
            lightcone redshift

        logmp_obs: ndarray of shape (n_z*n_m, )
            base-10 log of halo mass at the lightcone redshift, in Msun

        mah_params: namedtuple of ndarray's with shape (n_z*n_m, )
            diffmah parameters

        logmp0: ndarray of shape (n_z*n_m, )
            base-10 log of halo mass at z=0, in Msun

        nhalos: ndarray of shape (n_z*n_m, )
            number of halos of this mass and redshift
    """
    # NOTE: replace since it is using regular grid
    # nhalo_weighted_lc_grid = halo_lightcone_weights(
    #     lgmp_grid,
    #     z_grid,
    #     sky_area_degsq,
    #     hmf_params=hmf_params,
    #     cosmo_params=cosmo_params,
    # )
    nhalo_weighted_lc_grid = get_nhalo_weighted_lc_grid(
        lgmp_grid,
        z_grid,
        sky_area_degsq,
        hmf_params=hmf_params,
        cosmo_params=cosmo_params,
    )
    nhalo_weights = nhalo_weighted_lc_grid.flatten()
    z_obs = np.repeat(z_grid, lgmp_grid.size)
    logmp_obs_mf = np.tile(lgmp_grid, z_grid.size)

    t_obs = flat_wcdm.age_at_z(z_obs, *cosmo_params)
    t_0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t_0)

    logmp_obs_mf_clipped = np.clip(logmp_obs_mf, logmp_cutoff, logmp_cutoff_himass)

    tarr = np.array((10**lgt0,))
    args = (diffmahpop_params, tarr, logmp_obs_mf_clipped, t_obs, ran_key, lgt0)
    mah_params_uncorrected = mc_cenpop(*args)[0]  # mah_params, dmhdt, log_mah

    logmp_obs_orig = _log_mah_kern(mah_params_uncorrected, t_obs, lgt0)
    delta_logmh_clip = logmp_obs_orig - logmp_obs_mf
    mah_params = mah_params_uncorrected._replace(
        logm0=mah_params_uncorrected.logm0 - delta_logmh_clip
    )

    logmp0 = _log_mah_kern(mah_params, 10**lgt0, lgt0)
    logmp_obs = _log_mah_kern(mah_params, t_obs, lgt0)

    fields = ("z_obs", "t_obs", "logmp_obs", "mah_params", "logmp0")
    values = (z_obs, t_obs, logmp_obs, mah_params, logmp0)
    cenpop_out = dict()
    for key, value in zip(fields, values):
        cenpop_out[key] = value
    cenpop_out["nhalos"] = nhalo_weights

    return cenpop_out


def mc_weighted_halo_lightcone(
    ran_key,
    num_halos,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
):
    """
    Generate a weighted population of halos, with MAHs,
    on the a sobol grid with weighted halo counts;
    basically this function sets up and performs a call to
    ``get_weighted_lightcone_sobol_host_halo_diffmah``

    Parameters
    ----------
    ran_key: jran.key
        random key

    num_halos: int
        number of halos to generate

    z_min: float
        minimum redshift value

    z_max: float
        maximum redshift value

    lgmp_min: float
        minimum halo mass, in Msun

    lgmp_max: float
        maximum halo mass, in Msun

    sky_area_degsq: float
        sky area, in deg^2

    hmf_params: namedtuple
        halo mass function parameters

    diffmahpop_params: namedtuple
        diffmahpop parameters

    logmp_cutoff: float
        base-10 log of minimum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun;
        for logmp < logmp_cutoff, P(θ_MAH | logmp) = P(θ_MAH | logmp_cutoff)

    logmp_cutoff_himass: float
        base-10 log of maximum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun

    Returns
    -------
    res: dict with keys:
        z_obs: ndarray of shape (num_halos, )
            redshift values

        t_obs: ndarray of shape (num_halos, )
            cosmic time at observation, in Gyr

        logmp_obs: ndarray of shape (num_halos, )
            base-10 log of halo mass at observation, in Msun

        mah_params: namedtuple of ndarrays of shape (num_halos, )
            mah parameters

        logmp0: ndarray of shape (num_halos, )
            base-10 log of halo mass at z=0, in Msun

        nhalos: ndarray of shape (num_halos, )
            weighted number of halos at each grid point
    """
    ran_key, ran_key_sobol = jran.split(ran_key, 2)

    # Generate Sobol sequence for halo masses and redshifts
    seed = int(jran.randint(ran_key_sobol, (), 0, 2**31 - 1))
    bits = None
    if num_halos > 1e9:
        # 64-bit sequence required to generate over 2^30 halos
        bits = 64
    sampler = qmc.Sobol(d=2, scramble=True, rng=seed, bits=bits)

    with warnings.catch_warnings():
        # Ignore warning about Sobol sequences not being fully balanced
        warnings.filterwarnings("ignore", category=UserWarning)
        sample = sampler.random(num_halos)
    z_obs, logmp_obs_mf = qmc.scale(sample, (z_min, lgmp_min), (z_max, lgmp_max)).T

    mclh_args = (
        ran_key,
        z_obs,
        logmp_obs_mf,
        sky_area_degsq,
    )
    mclh_kwargs = dict(
        hmf_params=hmf_params,
        diffmahpop_params=diffmahpop_params,
        logmp_cutoff=logmp_cutoff,
        logmp_cutoff_himass=logmp_cutoff_himass,
    )

    # generate the halo population
    res = get_weighted_lightcone_host_halo_diffmah(
        *mclh_args, **mclh_kwargs
    )  # type: ignore

    return res


def get_weighted_lightcone_host_halo_diffmah(
    ran_key,
    z_obs,
    logmp_obs_mf,
    sky_area_degsq,
    cosmo_params=DEFAULT_COSMOLOGY,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
):
    """
    Generates a weighted lightcone population of halos with MAHs,
    on a given grid of redshift and mass

    Parameters
    ----------
    ran_key: jran.key
        random key

    z_obs: ndarray of shape (n_halo, )
        observed redshifts of galaxies

    logmp_obs_mf: ndarray of shape (n_halo, )
        base-10 log of observed halo masses, in Msun

    sky_area_degsq: float
        sky area, in deg^2

    cosmo_params: namedtuple
        cosmological parameters

    hmf_params: namedtuple
        halo mass function parameters

    diffmahpop_params: namedtuple
        diffmahpop parameters

    logmp_cutoff: float
        base-10 log of minimum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun;
        for logmp < logmp_cutoff, P(θ_MAH | logmp) = P(θ_MAH | logmp_cutoff)

    logmp_cutoff_himass: float
        base-10 log of maximum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun

    Returns
    -------
    cenpop_out: dict
        with keys:
        z_obs: ndarray of shape (n_halo, )
            redshift values

        t_obs: ndarray of shape (n_halo, )
            cosmic time at observation, in Gyr

        logmp_obs: ndarray of shape (n_halo, )
            base-10 log of halo mass at observation, in Msun

        mah_params: namedtuple of ndarrays of shape (n_halo, )
            mah parameters

        logmp0: ndarray of shape (n_halo, )
            base-10 log of halo mass at z=0, in Msun

        nhalos: ndarray of shape (n_halo, )
            weighted number of halos at each grid point
    """

    nhalo_weights = halo_lightcone_weights(
        logmp_obs_mf,
        z_obs,
        sky_area_degsq,
        hmf_params=hmf_params,
        cosmo_params=cosmo_params,
    )
    t_obs = flat_wcdm.age_at_z(z_obs, *cosmo_params)
    t_0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t_0)

    logmp_obs_mf_clipped = np.clip(logmp_obs_mf, logmp_cutoff, logmp_cutoff_himass)

    tarr = np.array((10**lgt0,))

    ran_key, mah_key = jran.split(ran_key, 2)
    args = (diffmahpop_params, tarr, logmp_obs_mf_clipped, t_obs, mah_key, lgt0)
    mah_params_uncorrected = mc_cenpop(*args)[0]

    logmp_obs_orig = _log_mah_kern(mah_params_uncorrected, t_obs, lgt0)
    delta_logmh_clip = logmp_obs_orig - logmp_obs_mf
    mah_params = mah_params_uncorrected._replace(
        logm0=mah_params_uncorrected.logm0 - delta_logmh_clip
    )

    logmp0 = _log_mah_kern(mah_params, 10**lgt0, lgt0)
    logmp_obs = _log_mah_kern(mah_params, t_obs, lgt0)

    fields = ("z_obs", "t_obs", "logmp_obs", "mah_params", "logmp0")
    values = (z_obs, t_obs, logmp_obs, mah_params, logmp0)
    cenpop_out = dict()
    for key, value in zip(fields, values):
        cenpop_out[key] = value
    cenpop_out["nhalos"] = nhalo_weights

    return cenpop_out
