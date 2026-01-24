"""Generate MC realizations of the host dark matter halo mass function"""

import numpy as np

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from .hmf_model import (
    DEFAULT_HMF_PARAMS,
    predict_cuml_hmf,
    _compute_nhalos_tot,
)

N_LGMU_TABLE = 200
U_TABLE = jnp.linspace(0, 1, N_LGMU_TABLE)
LGMH_MAX = 17.0

__all__ = ("mc_host_halos_singlez",)


def mc_host_halos_singlez(
    ran_key,
    lgmp_min,
    redshift,
    volume_com_mpc,
    hmf_params=DEFAULT_HMF_PARAMS,
    lgmp_max=LGMH_MAX,
):
    """
    Monte Carlo realization of the host halo mass function

    Parameters
    ----------
    ran_key: jran.PRNGKey
        random key

    lgmp_min: float
        base-10 log of the halo mass competeness limit of the
        generated population Halo mass is in units of Msun (not Msun/h);
        smaller values of lgmp_min produce more halos in the returned sample

    redshift: float
        redshift of the halo population

    volume_com_mpc: float
        comoving volume of the generated population, in Mpc^3

    hmf_params: namedtuple
        HMF parameters named tuple

    lgmp_max: float
        base-10 log of the maximum mass

    Returns
    -------
    lgmp_halopop: ndarray, shape (n_halos, )
        base-10 log of the halo mass of the generated population

    Notes
    -----
    Note that both number density and halo mass are defined in
    physical units (not h=1 units)
    """

    mean_nhalos = _compute_nhalos_tot(
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
    mean_nhalos = mean_nhalos - mean_nhalos_lgmax

    nhalos = jran.poisson(ran_key, mean_nhalos)

    uran = jran.uniform(ran_key, minval=0, maxval=1, shape=(nhalos,))
    lgmp_halopop = jnp.array(
        _mc_host_halos_singlez_kern(
            uran,
            hmf_params,
            lgmp_min,
            redshift,
            lgmp_max=lgmp_max,
        )
    )

    return lgmp_halopop


@jjit
def _get_hmf_cdf_interp_tables(
    hmf_params,
    lgmp_min,
    redshift,
    lgmp_max=LGMH_MAX,
):
    dlgmp = lgmp_max - lgmp_min
    lgmp_table = U_TABLE * dlgmp + lgmp_min

    cdf_table = 10 ** predict_cuml_hmf(hmf_params, lgmp_table, redshift)
    cdf_table = cdf_table - cdf_table[0]
    cdf_table = cdf_table / cdf_table[-1]

    return lgmp_table, cdf_table


@jjit
def _mc_host_halos_singlez_kern(
    uran,
    hmf_params,
    lgmp_min,
    redshift,
    lgmp_max=LGMH_MAX,
):
    lgmp_table, cdf_table = _get_hmf_cdf_interp_tables(
        hmf_params,
        lgmp_min,
        redshift,
        lgmp_max=lgmp_max,
    )
    mc_lg_mp = jnp.interp(uran, cdf_table, lgmp_table)

    return mc_lg_mp


def mc_host_halos_hist_singlez(
    ran_key,
    lgmp_min,
    redshift,
    volume_com_mpc,
    hmf_params=DEFAULT_HMF_PARAMS,
    lgmp_max=LGMH_MAX,
    n_bins=20,
):
    """
    Generate a histogram of a Monte Carlo realization of the
    host halo mass function at the input redshift

    Parameters
    ----------
    ran_key: jran.PRNGKey
        random key

    lgmp_min: float
        base-10 log of the halo mass competeness limit of the
        generated population Halo mass is in units of Msun (not Msun/h);
        smaller values of lgmp_min produce more halos in the returned sample

    redshift: float
        redshift of the halo population

    volume_com_mpc: float
        comoving volume of the generated population, in Mpc^3

    hmf_params: namedtuple
        HMF parameters named tuple

    lgmp_max: float
        base-10 log of the maximum mass

    n_bins: int
        number of histogram bins

    Returns
    -------
    dnhalo_bins: ndarray of shape (n_bins, )
        hmf bin counts of the generated population

    dlogm_bins: ndarray of shape (n_bins, )
        base-10 log of halo mass of halos in bins
    """
    mc_logmhalo = mc_host_halos_singlez(
        ran_key,
        lgmp_min,
        redshift,
        volume_com_mpc,
        hmf_params=hmf_params,
        lgmp_max=lgmp_max,
    )

    hist_data = np.histogram(mc_logmhalo, bins=n_bins, density=False)

    dlogm_bin_edges = hist_data[1]
    dlogm_bins = 0.5 * (dlogm_bin_edges[1:] + dlogm_bin_edges[:-1])

    n_halo = mc_logmhalo.size
    n_tot = _compute_nhalos_tot(
        hmf_params,
        lgmp_min,
        redshift,
        volume_com_mpc,
    )

    # normalize counts
    dnhalo_bins = hist_data[0] / np.diff(dlogm_bin_edges) / volume_com_mpc
    dnhalo_bins *= n_tot / n_halo

    return dnhalo_bins, dlogm_bins


def mc_host_halos_hist_singlez_out_of_core(
    ran_key,
    lgmp_min,
    redshift,
    volume_com_mpc_per_subvol,
    n_subvol,
    hmf_params=DEFAULT_HMF_PARAMS,
    lgmp_max=LGMH_MAX,
    n_bins=20,
):
    """
    Generate a histogram of a Monte Carlo realization of the
    host halo mass function at the input redshift,
    based on the ``mc_host_halos_hist_singlez`` function,
    using the out-of-core method to run in subvolumes to avoid
    overflowing memory, combining all subvolumes in the end

    Parameters
    ----------
    ran_key: jran.PRNGKey
        random key

    lgmp_min: float
        base-10 log of the halo mass competeness limit of the
        generated population Halo mass is in units of Msun (not Msun/h);
        smaller values of lgmp_min produce more halos in the returned sample

    redshift: float
        redshift of the halo population

    volume_com_mpc_per_subvol: float
        comoving volume per sublovume in units of Mpc^3

    n_subvol: int
        number of subvolumes to generate

    hmf_params: namedtuple
        HMF parameters named tuple

    lgmp_max: float
        base-10 log of the maximum mass

    n_bins: int
        number of histogram bins

    Returns
    -------
    halo_counts: ndarray, shape (n_bins, )
        halo counts of the generated population, in 1/Mpc^3/d(logMhalo)

    logm_bins: ndarray, shape (n_bins, )
        base-10 log of the halo mass bin centers
        of the generated population, in Msun
    """
    logm_bin_edges = np.linspace(lgmp_min, lgmp_max, n_bins + 1)
    halo_counts = np.zeros(n_bins)
    n_halo = 0

    for _ in range(n_subvol):
        mc_logmhalo = mc_host_halos_singlez(
            ran_key,
            lgmp_min,
            redshift,
            volume_com_mpc_per_subvol,
            hmf_params=hmf_params,
            lgmp_max=lgmp_max,
        )

        # put halo masses into bins
        lgmp_digitized = jnp.digitize(mc_logmhalo, logm_bin_edges)
        # counts of unique mass bins
        hist_index, unique_counts = jnp.unique_counts(lgmp_digitized)
        # put the counts into the appropriate histogram bin
        halo_counts[hist_index - 1] += unique_counts

        # keep count of the total number of generated halos for normalization
        n_halo += mc_logmhalo.size

        del mc_logmhalo

    # normalize counts
    n_tot = _compute_nhalos_tot(
        hmf_params,
        lgmp_min,
        redshift,
        volume_com_mpc_per_subvol * n_subvol,
    )
    halo_counts /= np.diff(logm_bin_edges)
    halo_counts /= volume_com_mpc_per_subvol * n_subvol
    halo_counts *= n_tot / n_halo

    # get the bin mid values to go with the counts
    logm_bins = 0.5 * (logm_bin_edges[1:] + logm_bin_edges[:-1])

    return halo_counts, logm_bins
