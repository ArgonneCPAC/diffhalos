""" """

import numpy as np

from ..utils import match_cenpop_to_subpop


def test_match_cenpop_to_subpop_computes():

    n_host = 100

    lgmhost_arr = np.linspace(10.0, 13.0, n_host)
    subhalo_counts_per_halo = (np.ones(n_host) * 5).astype(int)
    nsub_tot = int(subhalo_counts_per_halo.sum())

    lgmhost_pop, host_halo_indx = match_cenpop_to_subpop(
        lgmhost_arr,
        subhalo_counts_per_halo,
        nsub_tot,
    )

    assert np.all(np.isfinite(lgmhost_pop))
    assert np.all(np.isfinite(host_halo_indx))
    assert lgmhost_pop.size == n_host * 5
    assert lgmhost_pop[0] == lgmhost_arr[0]
    assert lgmhost_pop[-1] == lgmhost_arr[-1]
    assert host_halo_indx.size == n_host * 5
    assert host_halo_indx[0] == 0
    assert host_halo_indx[-1] == n_host - 1
