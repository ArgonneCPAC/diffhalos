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
    """Check each returned column is finite and has the expected shape"""
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

    n_subs = n_host * halopop.nsub_per_host
    n_tot = n_host + n_subs

    # Check mah_params for shape and finite
    assert len(halopop.mah_params) == 5
    for arr in halopop.mah_params:
        assert np.all(np.isfinite(arr))
        assert arr.shape == (n_tot,)

    # Check other columns for shape and finite
    skip_list = ("logt0", "mah_params", "nsub_per_host")
    for name, val in zip(halopop._fields, halopop):
        if name not in skip_list:
            assert val.shape == (n_tot,), name
            assert np.all(np.isfinite(val))

    # Check scalar columns for shape and finite
    assert halopop.logt0.shape == ()
    assert np.isfinite(halopop.logt0)
    assert halopop.nsub_per_host.shape == ()
    assert np.isfinite(halopop.nsub_per_host)


def test_weighted_lc_logmp0_is_consistent_with_logmp_obs():
    """Enforce self-consistent behavior for logmp0 and logmp_obs columns
    centrals: logmp_obs <= logmp0
    satellites: logmp_obs == logmp0

    """
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

    # centrals: logmp_obs <= logmp0
    assert np.all(halopop.logmp0[:n_host] >= halopop.logmp_obs[:n_host])

    # satellites: logmp_obs == logmp0
    assert np.allclose(halopop.logmp0[n_host:], halopop.logmp_obs[n_host:])


def test_weighted_lc_tpeak_subs():
    """Enforce self-consistent behavior for logmp0 and logmp_obs columns
    centrals: logmp_obs <= logmp0
    satellites: logmp_obs == logmp0

    """
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

    # satellites: t_peak <= t_obs
    assert np.all(halopop.mah_params.t_peak[n_host:] <= halopop.t_obs[n_host:])

    # at least SOME satellites should have t_peak != t_obs
    assert np.any(halopop.mah_params.t_peak[n_host:] < halopop.t_obs[n_host:])
