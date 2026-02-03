""" """

import numpy as np
from jax import random as jran

from ...ccshmf.utils import match_cenpop_to_subpop
from .. import mc_lightcone as mclc


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


def test_mc_lc_behaves_as_expected():
    ran_key = jran.key(0)

    lgmp_min = 12.0
    lgmsub_min = 10.0
    z_min = 1.2
    z_max = 1.5
    sky_area_degsq = 10.0

    halopop = mclc.mc_lc(
        ran_key,
        lgmp_min,
        lgmsub_min,
        z_min,
        z_max,
        sky_area_degsq,
    )

    for _field in halopop._fields:
        assert np.all(np.isfinite(halopop._asdict()[_field]))

    n_host = halopop.logmp_obs.size
    n_subs = halopop.logmu_obs.size
    for _param in halopop.mah_params._fields:
        assert halopop.mah_params._asdict()[_param].size == n_host + n_subs

    assert halopop.halo_indx.size == n_host + n_subs


def test_weighted_lc_behaves_as_expected():
    ran_key = jran.key(0)

    n_host = 100
    z_obs = np.linspace(0.2, 1.5, n_host)
    logmp_obs = np.linspace(11.0, 14.0, n_host)

    lgmsub_min = 10.0
    sky_area_degsq = 10.0

    halopop = mclc.weighted_lc(
        ran_key,
        z_obs,
        logmp_obs,
        lgmsub_min,
        sky_area_degsq,
    )

    for _field in halopop._fields:
        assert np.all(np.isfinite(halopop._asdict()[_field]))

    n_subs = n_host * halopop.nsub_per_host
    n_tot = n_host + n_subs
    for arr in halopop.mah_params:
        assert arr.size == n_tot

    assert halopop.halo_indx.size == n_tot

    assert halopop.nhalos.size == n_tot

    assert halopop.logmp_obs.shape == (n_tot,)
