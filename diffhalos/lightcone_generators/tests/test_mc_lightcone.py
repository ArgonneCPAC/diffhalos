""" """

import numpy as np
from diffmah.diffmah_kernels import _log_mah_kern
from jax import numpy as jnp
from jax import random as jran

from ...ccshmf.ccshmf_model import subhalo_lightcone_weights
from ...ccshmf.utils import match_cenpop_to_subpop
from ...mah.diffmahnet.diffmahnet import log_mah_kern
from ...mah.diffmahnet_utils import mc_mah_cenpop
from .. import mc_lightcone as mclc
from .. import mc_lightcone_halos as mclch
from .. import mc_lightcone_subhalos as mclcsh


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

    n_host_halos = 100
    z_min, z_max = 0.1, 3.1
    sky_area_degsq = 10.0
    lgmp_min, lgmp_max = 10.0, 15.0
    args = (ran_key, n_host_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area_degsq)
    halopop = mclc.weighted_lc(*args)

    n_subs = n_host_halos * halopop.nsub_per_host
    n_tot = n_host_halos + n_subs

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
    We perform these checks only for subhalos with t_peak < t_obs
    """
    ran_key = jran.key(0)

    n_host_halos = 100
    z_min, z_max = 0.1, 3.1
    sky_area_degsq = 10.0
    lgmp_min, lgmp_max = 10.0, 15.0
    args = (ran_key, n_host_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area_degsq)
    halopop = mclc.weighted_lc(*args)

    # centrals: logmp_obs <= logmp0
    assert np.all(halopop.logmp0[:n_host_halos] >= halopop.logmp_obs[:n_host_halos])

    # satellites: logmp_obs == logmp0
    _filter_t_peak_t_obs = np.where(
        halopop.t_obs[n_host_halos:] > halopop.mah_params.t_peak[n_host_halos:]
    )[0]
    assert np.allclose(
        halopop.logmp0[n_host_halos:][_filter_t_peak_t_obs],
        halopop.logmp_obs[n_host_halos:][_filter_t_peak_t_obs],
    )


def test_weighted_lc_tpeak_subs():
    """Enforce self-consistent behavior for t_peak and t_obs columns
    satellites: t_peak <= t_obs
    for at least some satellites: t_peak != t_obs
    """
    ran_key = jran.key(0)

    n_host_halos = 1000
    z_min, z_max = 0.1, 3.1
    sky_area_degsq = 10.0
    lgmp_min, lgmp_max = 10.0, 15.0
    args = (ran_key, n_host_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area_degsq)
    halopop = mclc.weighted_lc(*args)

    # satellites: t_peak <= t_obs
    # ensure that the fraction of subs for which this is false is < 10%
    _filter_t_peak_t_obs = np.where(
        halopop.t_obs[n_host_halos:] < halopop.mah_params.t_peak[n_host_halos:]
    )[0]
    tol = 0.15
    assert len(_filter_t_peak_t_obs) / float(halopop.nsub_per_host * n_host_halos) < tol

    # at least SOME satellites should have t_peak != t_obs
    assert np.any(
        halopop.mah_params.t_peak[n_host_halos:] != halopop.t_obs[n_host_halos:]
    )

    # make sure it is the same subhalos that have t_peak>t_obs and logm_obs!=logm0
    _filter_m_obs_m0 = np.where(
        halopop.logmp0[n_host_halos:] != halopop.logmp_obs[n_host_halos:]
    )[0]
    assert np.all(_filter_m_obs_m0 == _filter_t_peak_t_obs)


def test_weighted_lc_tpeak_clip():
    """Enforce self-consistent behavior for t_peak and t_obs columns
    satellites: t_peak <= t_obs
    by clipping t_peak at t_obs in the diffmahnet parameters
    """
    ran_key = jran.key(0)
    cen_key, sub_key = jran.split(ran_key, 2)

    n_host = 100
    z_obs = np.linspace(0.2, 1.5, n_host)
    logmp_obs = np.linspace(11.0, 14.0, n_host)

    lgmsub_min = 10.0
    sky_area_degsq = 10.0

    cenpop = mclch._weighted_lc_halos_from_grid(
        cen_key,
        z_obs,
        logmp_obs,
        sky_area_degsq,
    )

    # number of host halos
    n_host = cenpop.logmp_obs.size

    # get mu values
    lgmu = subhalo_lightcone_weights(
        ran_key,
        cenpop.logmp_obs,
        lgmsub_min,
        mclcsh.N_LGMU_PER_HOST,
        mclcsh.DEFAULT_CCSHMF_PARAMS,
    )[1].reshape(n_host * mclcsh.N_LGMU_PER_HOST)

    # get the subhalo mass and time of observation for MAH computations
    logmsub_obs = lgmu + jnp.repeat(cenpop.logmp_obs, mclcsh.N_LGMU_PER_HOST)
    t_obs = jnp.repeat(cenpop.t_obs, mclcsh.N_LGMU_PER_HOST)

    # get the rescaled mah parameters and mah values at t_obs
    logmsub_obs_clipped = jnp.clip(
        logmsub_obs, mclcsh.DEFAULT_LOGMSUB_CUTOFF, mclcsh.DEFAULT_LOGMSUB_HIMASS_CUTOFF
    )
    mah_params_uncorrected = mc_mah_cenpop(
        logmsub_obs_clipped,
        t_obs,
        sub_key,
        mclcsh.DEFAULT_DIFFMAHNET_SAT_MODEL,
    )

    # clip t_peak values for subs's mah parameters
    t_peak_clip = jnp.clip(mah_params_uncorrected.t_peak, 0.0, t_obs)
    mah_params_uncorrected = mah_params_uncorrected._replace(t_peak=t_peak_clip)

    # compute the uncorrected observed masses
    logmp_obs_uncorrected = log_mah_kern(mah_params_uncorrected, t_obs, cenpop.logt0)

    # rescale the mah parameters to the correct logm0
    delta_logm_obs = logmp_obs_uncorrected - logmsub_obs
    logm0_rescaled = mah_params_uncorrected.logm0 - delta_logm_obs
    mah_params_subs = mah_params_uncorrected._replace(logm0=logm0_rescaled)

    # compute observed mass with corrected parameters
    logmsub_obs = log_mah_kern(mah_params_subs, t_obs, cenpop.logt0)

    # compute mass of subs at z=0
    logmp0_subs = _log_mah_kern(mah_params_subs, 10**cenpop.logt0, cenpop.logt0)

    # satellites: t_peak <= t_obs
    assert np.all(mah_params_subs.t_peak <= t_obs)

    # at least SOME satellites should have t_peak != t_obs
    assert np.any(mah_params_subs.t_peak != t_obs)

    # satellites: logmp_obs == logmp0
    assert np.allclose(logmp0_subs, logmsub_obs)


def test_weighted_lc_nhalos_host():
    ran_key = jran.key(0)

    n_host_halos = 100
    z_min, z_max = 0.1, 3.1
    sky_area_degsq = 10.0
    lgmp_min, lgmp_max = 10.0, 15.0
    args = (ran_key, n_host_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area_degsq)
    halopop = mclc.weighted_lc(*args)

    assert np.allclose(halopop.central[:n_host_halos], 1)
    assert np.allclose(halopop.central[n_host_halos:], 0)

    assert np.allclose(halopop.nhalos_host[:n_host_halos], 1)

    assert np.allclose(
        halopop.nhalos_host[n_host_halos:],
        halopop.nhalos[halopop.halo_indx][n_host_halos:],
    )
