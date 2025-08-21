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
    lgmhost_arr,
    lgmp_min,
    volume_com_mpc,
    ccshmf_params=DEFAULT_CCSHMF_PARAMS,
    n_bins=20,
    bin_in="logmu",
):
    """
    Generate a histogram of a population of subhalos
    with synthetic values of Mpeak

    Parameters
    ----------
    ran_key: jax.random.PRNGKey
        random key

    lgmhost_arr: ndarray of shape (nhosts, )
        base-10 log of host halo mass, in Msun

    lgmp_min: float
        base-10 log of the smallest Mpeak value
        of the synthetic subhalos, in Msun

    volume_com_mpc: float
        comoving volume of the generated population in units of Mpc^3;
        larger values of volume_com produce more halos in the returned sample

    cshmf_params: namedtuple
        CCSHMF parameters named tuple

    n_bins: int
        number of histogram bins

    bin_in: str
        quantity to bin in;
        options:
        - "logmu": log10(mu)=log10(Msub/Mhost)
        - "logmhost": log10(M_host)

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
    mc_lg_mu, lgmhost_pop, _ = generate_subhalopop(
        ran_key,
        lgmhost_arr,
        lgmp_min,
        ccshmf_params=ccshmf_params,
    )

    if bin_in == "logmu":
        x = mc_lg_mu
    elif bin_in == "logmhost":
        x = lgmhost_pop
    else:
        print("!ERROR! Input %s for `bin_in` is not valid" % bin_in)
        errmsg = "Choose from ['logmu', 'logmhost']"
        raise Exception(errmsg)

    hist_data = np.histogram(x, bins=n_bins, density=False)
    dx_bin_edges = hist_data[1]
    dx_bins = 0.5 * (dx_bin_edges[1:] + dx_bin_edges[:-1])

    n_sub = x.size
    mean_counts = _compute_mean_subhalo_counts(lgmhost_arr, lgmp_min)
    _, counts_key = jran.split(ran_key, 2)
    subhalo_counts_per_halo = jran.poisson(counts_key, mean_counts)
    n_tot = jnp.sum(subhalo_counts_per_halo)

    # normalize counts
    dsub_bins = hist_data[0] / np.diff(dx_bin_edges) / volume_com_mpc
    dsub_bins *= n_tot / n_sub

    return dsub_bins, dx_bins


def get_lgmu_cutoff(lgmhost, lgmp_sim, nptcl_cut):
    lgmp_cutoff = lgmp_sim + np.log10(nptcl_cut)
    lgmu_cutoff = lgmp_cutoff - lgmhost
    return lgmu_cutoff


def mc_generate_subhalopop_singlehalo(
    ran_key, lgmu_table, ntot, ccshmf_kern_params=DEFAULT_CCSHMF_KERN_PARAMS
):
    uran = jran.uniform(ran_key, minval=0, maxval=1, shape=(ntot,))
    cdf_counts = 10 ** lg_ccshmf_kern(ccshmf_kern_params, lgmu_table)
    cdf_counts = cdf_counts - cdf_counts[0]
    cdf_counts = cdf_counts / cdf_counts[-1]

    mc_lg_mu = np.interp(uran, cdf_counts, lgmu_table)

    return mc_lg_mu


@jjit
def generate_subhalopop_kern(
    uran, lgmhost, lgmp_min, ccshmf_params=DEFAULT_CCSHMF_PARAMS
):
    lgmu_cutoff = get_lgmu_cutoff(lgmhost, lgmp_min, 1)
    lgmu_table = U_TABLE * lgmu_cutoff
    cdf_counts = 10 ** predict_ccshmf(ccshmf_params, lgmhost, lgmu_table)
    cdf_counts = cdf_counts - cdf_counts[0]
    cdf_counts = cdf_counts / cdf_counts[-1]

    mc_lg_mu = jnp.interp(uran, cdf_counts, lgmu_table)

    return mc_lg_mu


_A = (0, 0, None, None)
generate_subhalopop_vmap = jjit(vmap(generate_subhalopop_kern, in_axes=_A))


def _compute_mean_subhalo_counts(
    lgmhost, lgmp_min, ccshmf_params=DEFAULT_CCSHMF_PARAMS
):
    lgmu_cutoff = get_lgmu_cutoff(lgmhost, lgmp_min, 1)
    mean_counts = 10 ** predict_ccshmf(ccshmf_params, lgmhost, lgmu_cutoff)
    return mean_counts
