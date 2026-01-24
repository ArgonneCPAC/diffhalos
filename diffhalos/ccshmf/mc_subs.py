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

from .ccshmf_model import (
    DEFAULT_CCSHMF_PARAMS,
    predict_cuml_cshmf,
    get_lgmu_cutoff,
    compute_mean_subhalo_counts,
)

N_LGMU_TABLE = 100
U_TABLE = np.linspace(1, 0, N_LGMU_TABLE)

__all__ = ("generate_subhalopop",)


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
    mc_lg_mu: ndarray of shape (n_mu, )
        base-10 log of mu=Msub/Mhost of the Monte Carlo subhalo population

    lgmhost_pop: ndarray of shape (n_mu*n_host, )
        base-10 log of Mhost of the Monte Carlo subhalo population, in Msun

    host_halo_indx: ndarray of shape (n_mu*n_host, )
        index of the input host halo of each generated subhalo,
        so that lgmhost_pop = lgmhost_arr[host_halo_indx];
        thus all values satisfy 0 <= host_halo_indx < nhosts
    """

    # random keys for uniforms randoms, and subhalo counts
    uran_key, counts_key = jran.split(ran_key, 2)

    # generate a Poisson realization of subhalo counts per host halo
    mean_counts = compute_mean_subhalo_counts(lgmhost_arr, lgmp_min)
    subhalo_counts_per_halo = jran.poisson(counts_key, mean_counts)

    # compute the total number of subhalos
    nsub_tot = jnp.sum(subhalo_counts_per_halo)

    # uniform randoms for mu sampling
    urandoms = jran.uniform(uran_key, shape=(nsub_tot,))

    # host halo population that matches the subhalo sample
    lgmhost_pop = jnp.repeat(
        lgmhost_arr,
        subhalo_counts_per_halo,
        total_repeat_length=nsub_tot,
    )
    halo_ids = jnp.arange(lgmhost_arr.size).astype(int)
    host_halo_indx = jnp.repeat(
        halo_ids,
        subhalo_counts_per_halo,
        total_repeat_length=nsub_tot,
    )

    # sample mu values for subhalos
    mc_lg_mu = generate_subhalopop_vmap(
        urandoms,
        lgmhost_pop,
        lgmp_min,
        ccshmf_params,
    )

    return mc_lg_mu, lgmhost_pop, host_halo_indx, subhalo_counts_per_halo


@jjit
def generate_subhalopop_kern(
    uran,
    lgmhost,
    lgmp_min,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
):
    """
    Kernel to generate a population of subhalos,
    for a single host-halo mass, and given a minimum cutoff halo mass

    Parameters
    ----------
    uran: ndarray of shape (n_subs, )
        uniform random numbers for sampling from
        the CDF of subhalo counts

    lgmhost: float
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
    cdf_counts = 10 ** predict_cuml_cshmf(ccshmf_params, lgmhost, lgmu_table)
    cdf_counts = cdf_counts - cdf_counts[0]
    cdf_counts = cdf_counts / cdf_counts[-1]

    mc_lg_mu = jnp.interp(uran, cdf_counts, lgmu_table)

    return mc_lg_mu


"""vmap function of subhalo generation within a host halo
for vectorized computations for multiple host halos simultaneously"""
_A = (0, 0, None, None)
generate_subhalopop_vmap = jjit(vmap(generate_subhalopop_kern, in_axes=_A))


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
    mean_counts = compute_mean_subhalo_counts(
        lgmhost, lgmp_min, ccshmf_params=ccshmf_params
    )
    uran_key, counts_key = jran.split(ran_key, 2)
    subhalo_counts_per_halo = jran.poisson(counts_key, mean_counts)
    n_tot = jnp.sum(subhalo_counts_per_halo)

    dnsub_bins = dnsub_bins / np.abs(np.diff(dlogmu_bin_edges))
    dnsub_bins *= n_tot / n_subhalo

    return dnsub_bins, dlogmu_bins
