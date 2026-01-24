""" """

import numpy as np
from jax import random as jran

from ..mc_subs import (
    generate_subhalopop_kern,
    get_mean_subhalo_counts_poisson,
    generate_subhalopop,
    DEFAULT_CCSHMF_PARAMS,
    get_lgmu_cutoff,
)


def test_generate_subhalopop_kern_behaves_as_expected():

    ran_key = jran.key(0)

    lgmhost = 10.0
    lgmp_min = 8.0

    nsubs = 100
    uran = jran.uniform(ran_key, minval=0, maxval=1, shape=(nsubs,))

    lgmu_subpop = generate_subhalopop_kern(
        uran,
        lgmhost,
        lgmp_min,
        ccshmf_params=DEFAULT_CCSHMF_PARAMS,
    )

    lgmu_cutoff = get_lgmu_cutoff(lgmhost, lgmp_min, 1)

    assert np.all(np.isfinite(lgmu_subpop))
    assert lgmu_subpop.size == nsubs
    assert np.all((lgmu_subpop > lgmu_cutoff) * (lgmu_subpop < 0.0))


def test_get_mean_subhalo_counts_poisson_behaves_as_expected():

    ran_key = jran.key(0)

    lgmp_min = 8.0

    nhost = 100
    lgmhost_arr = np.linspace(10.0, 12.0, nhost)

    subhalo_counts_per_halo = get_mean_subhalo_counts_poisson(
        ran_key,
        lgmhost_arr,
        lgmp_min,
    )

    assert np.all(np.isfinite(subhalo_counts_per_halo))
    assert subhalo_counts_per_halo.size == nhost


def test_generate_subhalopop_behaves_as_expected():

    ran_key = jran.key(0)

    lgmp_min = 8.0

    nhost = 100
    lgmhost_arr = np.linspace(10.0, 12.0, nhost)

    mc_lg_mu, lgmhost_pop, host_halo_indx = generate_subhalopop(
        ran_key,
        lgmhost_arr,
        lgmp_min,
        ccshmf_params=DEFAULT_CCSHMF_PARAMS,
    )

    assert np.all(np.isfinite(mc_lg_mu))
    assert np.all(np.isfinite(lgmhost_pop))
    assert mc_lg_mu.size == lgmhost_pop.size == host_halo_indx.size
    assert host_halo_indx[-1] == lgmhost_arr.size - 1
