""" """

import numpy as np

from jax import random as jran

from .. import mc_lightcone as mclc
from ...ccshmf.utils import match_cenpop_to_subpop


def test_mc_lc_mf_behaves_as_expected():

    ran_key = jran.key(0)

    lgmp_min = 12.0
    lgmsub_min = 10.0
    z_min = 1.2
    z_max = 1.5
    sky_area_degsq = 10.0

    z_halopop, logmp_cenpop, mc_lg_mu, subhalo_counts_per_halo = mclc.mc_lc_mf(
        ran_key,
        lgmp_min,
        lgmsub_min,
        z_min,
        z_max,
        sky_area_degsq,
    )

    assert np.all(np.isfinite(z_halopop))
    assert np.all(np.isfinite(logmp_cenpop))
    assert np.all(np.isfinite(mc_lg_mu))
    assert np.all(np.isfinite(subhalo_counts_per_halo))
    assert z_halopop.size == logmp_cenpop.size
    assert mc_lg_mu.size == subhalo_counts_per_halo.sum()

    nsub_tot = int(subhalo_counts_per_halo.sum())
    lgmhost_pop, host_halo_indx = match_cenpop_to_subpop(
        logmp_cenpop,
        subhalo_counts_per_halo,
        nsub_tot,
    )
    assert lgmhost_pop.size == subhalo_counts_per_halo.sum()
    assert host_halo_indx.size == subhalo_counts_per_halo.sum()
    assert lgmhost_pop[0] == logmp_cenpop[0]
    assert lgmhost_pop[-1] == logmp_cenpop[-1]
    assert host_halo_indx[-1] == logmp_cenpop.size - 1
