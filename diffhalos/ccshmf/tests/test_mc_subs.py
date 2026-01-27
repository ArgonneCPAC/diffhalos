""" """

import numpy as np
import warnings

from jax import random as jran
from jax import numpy as jnp

from ..ccshmf_model import predict_diff_cshmf

from ..mc_subs import (
    generate_subhalopop_kern,
    generate_subhalopop_vmap,
    generate_subhalopop,
    DEFAULT_CCSHMF_PARAMS,
    get_lgmu_cutoff,
    compute_mean_subhalo_counts,
    generate_subhalopop_hist,
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


def test_generate_subhalopop_vmap_behaves_as_expected():

    ran_key = jran.key(0)
    ran_key, counts_key = jran.split(ran_key, 2)

    lgmhost_arr = np.linspace(10.0, 13.0, 100)
    lgmp_min = 8.0

    mean_counts = compute_mean_subhalo_counts(lgmhost_arr, lgmp_min)
    subhalo_counts_per_halo = jran.poisson(counts_key, mean_counts)
    nsubs = jnp.sum(subhalo_counts_per_halo)

    lgmhost_pop = jnp.repeat(
        lgmhost_arr,
        subhalo_counts_per_halo,
        total_repeat_length=nsubs,
    )

    uran = jran.uniform(ran_key, minval=0, maxval=1, shape=(nsubs,))

    lgmu_subpop = generate_subhalopop_vmap(
        uran,
        lgmhost_pop,
        lgmp_min,
        DEFAULT_CCSHMF_PARAMS,
    )

    lgmu_cutoff = get_lgmu_cutoff(lgmhost_pop, lgmp_min, 1)

    assert np.all(np.isfinite(lgmu_subpop))
    assert lgmu_subpop.size == nsubs
    assert np.all((lgmu_subpop > lgmu_cutoff) * (lgmu_subpop < 0.0))


def test_generate_subhalopop_behaves_as_expected():

    ran_key = jran.key(0)

    lgmp_min = 8.0

    nhost = 100
    lgmhost_arr = np.linspace(10.0, 12.0, nhost)

    mc_lg_mu, lgmhost_pop, host_halo_indx, subhalo_counts_per_halo = (
        generate_subhalopop(
            ran_key,
            lgmhost_arr,
            lgmp_min,
            ccshmf_params=DEFAULT_CCSHMF_PARAMS,
        )
    )

    assert np.all(np.isfinite(mc_lg_mu))
    assert np.all(np.isfinite(lgmhost_pop))
    assert mc_lg_mu.size == lgmhost_pop.size == host_halo_indx.size
    assert host_halo_indx[-1] == lgmhost_arr.size - 1
    assert np.all(np.isfinite(subhalo_counts_per_halo))
    assert subhalo_counts_per_halo.sum() == mc_lg_mu.size


def test_generate_subhalopop_hist_vs_theory_comparison():
    ran_key = jran.key(0)

    n_bins = 20
    lgmp_min = 7.0
    lgmhost_arr = np.linspace(10.0, 13.0, 5)

    for lgmhost in lgmhost_arr:
        cshmf_pop_hist, logmu_pop = generate_subhalopop_hist(
            ran_key,
            lgmhost,
            lgmp_min,
            ccshmf_params=DEFAULT_CCSHMF_PARAMS,
            bins=n_bins,
        )

        cshmf_pop_theory = predict_diff_cshmf(
            DEFAULT_CCSHMF_PARAMS,
            lgmhost,
            logmu_pop,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            _filter = np.where(logmu_pop < -2.0)[0]

            assert np.allclose(
                np.log10(cshmf_pop_hist)[_filter],
                cshmf_pop_theory[_filter],
                rtol=0.1,
            )
