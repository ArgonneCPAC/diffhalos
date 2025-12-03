"""
The generate_subhalopop function generates a Monte Carlo
realization of a subhalo population defined by its
cumulative conditional subhalo mass function, CCSHMF.
Starting with a simulated snapshot or lightcone with only host halos,
generate_subhalopop can be used to add subhalos with synthetic values of Mpeak.
"""

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from .ccshmf_model import DEFAULT_CCSHMF_PARAMS, predict_ccshmf
from .ccshmf_kernels import DEFAULT_CCSHMF_KERN_PARAMS, lg_ccshmf_kern

N_LGMU_TABLE = 100
U_TABLE = np.linspace(1, 0, N_LGMU_TABLE)

__all__ = (
    "generate_subhalopop",
    "generate_subhalopop_hist",
    "generate_subhalopop_hist_out_of_core",
)


def generate_subhalopop(
    ran_key,
    lgmhost_arr,
    lgmp_min,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
):
    """
    Generate a population of subhalos with synthetic values of Mpeak

    Parameters
    ----------
    ran_key: jax.random.PRNGKey
        random key

    lgmhost_arr: ndarray of shape (nhosts, )
        base-10 log of host halo mass, in Msun

    lgmp_min: float
        base-10 log of the smallest Mpeak value
        of the synthetic subhalos, in Msun

    cshmf_params: namedtuple
        CCSHMF parameters named tuple

    Returns
    -------
    mc_lg_mu: ndarray of shape (nsubs, )
        base-10 log of mu=Msub/Mhost of the Monte Carlo subhalo population

    lgmhost_pop: ndarray of shape (nsubs, )
        base-10 log of Mhost of the Monte Carlo subhalo population, in Msun

    host_halo_indx: ndarray of shape (nsubs, )
        index of the input host halo of each generated subhalo,
        so that lgmhost_pop = lgmhost_arr[host_halo_indx];
        thus all values satisfy 0 <= host_halo_indx < nhosts
    """
    mean_counts = _compute_mean_subhalo_counts(lgmhost_arr, lgmp_min)
    uran_key, counts_key = jran.split(ran_key, 2)
    subhalo_counts_per_halo = jran.poisson(counts_key, mean_counts)
    ntot = jnp.sum(subhalo_counts_per_halo)
    urandoms = jran.uniform(uran_key, shape=(ntot,))
    lgmhost_pop = np.repeat(lgmhost_arr, subhalo_counts_per_halo)
    halo_ids = np.arange(lgmhost_arr.size).astype(int)
    host_halo_indx = np.repeat(halo_ids, subhalo_counts_per_halo)
    mc_lg_mu = generate_subhalopop_vmap(
        urandoms,
        lgmhost_pop,
        lgmp_min,
        ccshmf_params,
    )
    return mc_lg_mu, lgmhost_pop, host_halo_indx


def generate_subhalopop_hist(
    ran_key,
    lgmhost,
    lgmp_min,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
    n_bins=20,
):
    """
    Generate a histogram of a population of subhalos
    with synthetic values of Mpeak

    Parameters
    ----------
    ran_key: jax.random.PRNGKey
        random key

    lgmhost: float
        base-10 log of host halo mass, in Msun

    lgmp_min: float
        base-10 log of the smallest Mpeak value
        of the synthetic subhalos, in Msun

    ccshmf_params: namedtuple
        CCSHMF parameters named tuple

    n_bins: int
        number of histogram bins

    Returns
    -------
    dsub_bins: ndarray of shape (n_bins, )
        binned subhalo counts

    dlogmu_bins: ndarray of shape (n_bins, )
        binned mu values
    """
    mc_lg_mu = generate_subhalopop(
        ran_key,
        lgmhost,
        lgmp_min,
        ccshmf_params=ccshmf_params,
    )[0]

    hist_data = np.histogram(mc_lg_mu, bins=n_bins, density=False)
    dlogmu_bin_edges = hist_data[1]
    dlogmu_bins = 0.5 * (dlogmu_bin_edges[1:] + dlogmu_bin_edges[:-1])

    # normalize counts
    dnsub_bins = hist_data[0] / np.diff(dlogmu_bin_edges)

    return dnsub_bins, dlogmu_bins


def generate_subhalopop_hist_out_of_core(
    ran_key,
    lgmhost,
    lgmp_min,
    n_real,
    logmu_min=-5.0,
    logmu_max=0.0,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
    n_bins=20,
):
    """
    Generate a histogram of a population of subhalos
    with synthetic values of Mpeak

    Parameters
    ----------
    ran_key: jax.random.PRNGKey
        random key

    lgmhost: float
        base-10 log of host halo mass, in Msun

    lgmp_min: float
        base-10 log of the smallest Mpeak value
        of the synthetic subhalos, in Msun

    n_real: int
        number of realizations to generate

    ccshmf_params: namedtuple
        CCSHMF parameters named tuple

    n_bins: int
        number of histogram bins

    Returns
    -------
    dsub_bins: ndarray of shape (n_bins, )
        binned subhalo counts

    dlogmu_bins: ndarray of shape (n_bins, )
        binned mu values
    """
    dlogmu_bin_edges = np.linspace(logmu_min, logmu_max, n_bins + 1)
    dnsub_bins = np.zeros(n_bins)
    n_subhalo = 0

    for i in range(n_real):
        mc_lg_mu = generate_subhalopop(
            ran_key,
            lgmhost,
            lgmp_min,
            ccshmf_params=ccshmf_params,
        )[0]

        # put mu values into bins
        lgmu_digitized = jnp.digitize(mc_lg_mu, dlogmu_bin_edges)
        # counts of unique mu bins
        hist_index, unique_counts = jnp.unique_counts(lgmu_digitized)
        # put the counts into the appropriate histogram bin
        dnsub_bins[hist_index - 1] += unique_counts

        # keep count of the total number of generated halos for normalization
        n_subhalo += mc_lg_mu.size

        del mc_lg_mu

    dlogmu_bins = 0.5 * (dlogmu_bin_edges[1:] + dlogmu_bin_edges[:-1])

    # normalize counts
    mean_counts = _compute_mean_subhalo_counts(
        lgmhost, lgmp_min, ccshmf_params=ccshmf_params
    )
    uran_key, counts_key = jran.split(ran_key, 2)
    subhalo_counts_per_halo = jran.poisson(counts_key, mean_counts)
    n_tot = jnp.sum(subhalo_counts_per_halo)

    dnsub_bins = dnsub_bins / np.abs(np.diff(dlogmu_bin_edges))
    dnsub_bins *= n_tot / n_subhalo

    return dnsub_bins, dlogmu_bins


def get_lgmu_cutoff(lgmhost, lgmp_sim, nptcl_cut):
    """
    Get the cutoff mu value for a simulation
    """
    lgmp_cutoff = lgmp_sim + np.log10(nptcl_cut)
    lgmu_cutoff = lgmp_cutoff - lgmhost
    return lgmu_cutoff


def mc_generate_subhalopop_singlehalo(
    ran_key,
    lgmu_table,
    ntot,
    ccshmf_kern_params=DEFAULT_CCSHMF_KERN_PARAMS,
):
    """
    Monte Carlo realization of a population of subhalos
    within a single host halo

    Parameters
    ----------
    ran_key: jax.random.PRNGKey
        random key

    lgmu_table: ndarray of shape (n_subs, )
        base-10 log of mu values for interpolation

    ntot: int
        total number of subhalos to generate

    ccshmf_params: namedtuple
        CCSHMF parameters named tuple

    Returns
    -------
    mc_lg_mu: ndarray of shape (nsubs, )
        base-10 log of mu=Msub/Mhost of the Monte Carlo subhalo population
    """
    uran = jran.uniform(ran_key, minval=0, maxval=1, shape=(ntot,))
    cdf_counts = 10 ** lg_ccshmf_kern(ccshmf_kern_params, lgmu_table)
    cdf_counts = cdf_counts - cdf_counts[0]
    cdf_counts = cdf_counts / cdf_counts[-1]

    mc_lg_mu = np.interp(uran, cdf_counts, lgmu_table)

    return mc_lg_mu


@jjit
def generate_subhalopop_kern(
    uran,
    lgmhost,
    lgmp_min,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
):
    """
    Kernel to generate a population of subhalos,
    given a minimum cutoff halo mass

    Parameters
    ----------
    uran: ndarray of shape (n_sub, )
        uniform random numbers for sampling from
        the CDF of subhalo counts

    lgmhost: ndarray of shape (nhosts, )
        base-10 log of host halo mass, in Msun

    lgmp_min: float
        base-10 log of the smallest Mpeak value
        of the synthetic subhalos, in Msun

    ccshmf_params: namedtuple
        CCSHMF parameters named tuple

    Returns
    -------
    mc_lg_mu: ndarray of shape (nsubs, )
        base-10 log of mu=Msub/Mhost of the Monte Carlo subhalo population
    """
    lgmu_cutoff = get_lgmu_cutoff(lgmhost, lgmp_min, 1)
    lgmu_table = U_TABLE * lgmu_cutoff
    cdf_counts = 10 ** predict_ccshmf(ccshmf_params, lgmhost, lgmu_table)
    cdf_counts = cdf_counts - cdf_counts[0]
    cdf_counts = cdf_counts / cdf_counts[-1]

    mc_lg_mu = jnp.interp(uran, cdf_counts, lgmu_table)

    return mc_lg_mu


"""vmap function of subhalo generation within a host halo
for vectorized computations for multiple host halos simultaneously"""
_A = (0, 0, None, None)
generate_subhalopop_vmap = jjit(vmap(generate_subhalopop_kern, in_axes=_A))


def _compute_mean_subhalo_counts(
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
