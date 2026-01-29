"""Useful utilities to be used with the lightcone module"""

from functools import partial

from jax import numpy as jnp
from jax import jit as jjit


__all__ = ("match_cenpop_to_subpop",)


@partial(jjit, static_argnames=["nsub_tot"])
def match_cenpop_to_subpop(
    lgmhost_arr,
    subhalo_counts_per_halo,
    nsub_tot,
):
    """
    Modify the host halo population properties to match
    the shape of the suhbalo population

    Parameters
    ----------
    lgmhost_arr: ndarray of shape (n_host, )
        base-10 log of host halo masses, in Msun

    subhalo_counts_per_halo: ndarray of shape (n_host, )
        number of subhalos generated per host halo

    nsub_tot: int
        total number of subhalos, subhalo_counts_per_halo.sum()

    Returns
    -------
    lgmhost_pop: ndarray of shape (n_hostn_host*subhalo_counts_per_halo, )
        base-10 log of Mhost of the Monte Carlo subhalo population, in Msun

    host_halo_indx: ndarray of shape (n_host*subhalo_counts_per_halo, )
        index of the input host halo of each generated subhalo,
        so that lgmhost_pop = lgmhost_arr[host_halo_indx]
    """
    # repeat host halo masses depending on number of subhalos per host
    lgmhost_pop = jnp.repeat(
        lgmhost_arr,
        subhalo_counts_per_halo,
        total_repeat_length=nsub_tot,
    )

    # assign indices to host halos that match them to the subhalos
    halo_ids = jnp.arange(lgmhost_arr.size).astype(int)
    host_halo_indx = jnp.repeat(
        halo_ids,
        subhalo_counts_per_halo,
        total_repeat_length=nsub_tot,
    )

    return lgmhost_pop, host_halo_indx
