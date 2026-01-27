""" """

import numpy as np

from jax import random as jran
from jax import numpy as jnp

from .. import mc_lightcone_subhalos as mclsh
from ..utils import generate_mock_cenpop
from ...ccshmf import mc_subs
from ...mah.utils import apply_mah_rescaling


def test_mc_lc_shmf_works_as_expected():
    ran_key = jran.key(0)

    n_tests = 5

    nhost = 100
    lgmhost = np.linspace(13.0, 15.0, nhost)
    lgmp_min = 12.0

    test_keys = jran.split(ran_key, n_tests)
    for test_key in test_keys:
        mc_lg_mu, lgmhost_pop, host_halo_indx, subs_per_halo = mclsh.mc_lc_shmf(
            test_key,
            lgmhost,
            lgmp_min,
        )

        assert mc_lg_mu.size == lgmhost_pop.size == host_halo_indx.size
        assert np.all(np.isfinite(mc_lg_mu))
        assert np.all(np.isfinite(lgmhost_pop))
        assert np.all(np.isfinite(host_halo_indx))
        assert np.all(np.isfinite(subs_per_halo))
        assert subs_per_halo.sum() == mc_lg_mu.size


def test_mc_lc_shmf_lgmp_min_parameter_behavior():
    ran_key = jran.key(0)

    n_tests = 5
    lgmp_mins = np.linspace(9.0, 11.0, n_tests)

    nhost = 100
    lgmhost = np.linspace(11.5, 14.0, nhost)

    for lgmp_min in lgmp_mins:
        mc_lg_mu, lgmhost_pop, host_halo_indx, subs_per_halo = mclsh.mc_lc_shmf(
            ran_key,
            lgmhost,
            lgmp_min,
        )

        assert mc_lg_mu.size == lgmhost_pop.size == host_halo_indx.size
        assert np.all(np.isfinite(mc_lg_mu))
        assert np.all(np.isfinite(lgmhost_pop))
        assert np.all(np.isfinite(host_halo_indx))
        assert np.all(np.isfinite(subs_per_halo))
        assert subs_per_halo.sum() == mc_lg_mu.size


def test_mc_lc_subhalos_works_as_expected():

    ran_key = jran.key(0)

    z_min = 0.2
    z_max = 2.0
    logmp_min = 11.0
    logmp_max = 14.0
    lgmp_min = 10.0
    n_cens = 1_000

    # use a mock host halo population for this test
    cenpop = generate_mock_cenpop(z_min, z_max, logmp_min, logmp_max, n_cens=n_cens)

    halopop = mclsh.mc_lc_subhalos(ran_key, cenpop, lgmp_min)

    subhalo_fields = ("host_index_for_subs", "mah_params_subs", "logmu_obs")

    for _field in subhalo_fields:
        assert _field in halopop._fields
        assert np.all(np.isfinite(halopop._asdict()[_field]))


def test_mc_lc_subhalos_vs_subpop_from_mc_subs():

    ran_key = jran.key(0)

    z_min = 0.4
    z_max = 0.5
    logmp_min = 11.0
    logmp_max = 14.0
    lgmp_min = 10.0
    n_cens = 2_000

    # use a mock host halo population for this test
    cenpop = generate_mock_cenpop(z_min, z_max, logmp_min, logmp_max, n_cens=n_cens)

    n_tests = 5
    test_keys = jran.split(ran_key, n_tests)
    for test_key in test_keys:
        lc_subpop = mclsh.mc_lc_subhalos(test_key, cenpop, lgmp_min)
        mc_lg_mu_pop = mc_subs.generate_subhalopop(
            test_key,
            cenpop.logmp_obs,
            lgmp_min,
        )[0]

        lg_mu_lc, lg_mu_bins = np.histogram(lc_subpop.logmu_obs, bins=100)
        lg_mu_pop, _ = np.histogram(mc_lg_mu_pop, bins=lg_mu_bins)
        msk_counts = lg_mu_pop > 500

        fracdiff = (lg_mu_lc[msk_counts] - lg_mu_pop[msk_counts]) / lg_mu_pop[
            msk_counts
        ]
        assert np.all(np.abs(fracdiff) < 0.2)

        del lc_subpop, mc_lg_mu_pop


def test_mah_params_rescaling():
    """Assert that the rescaled MAH parameters produce
    masses that are close to the expected ones"""

    ran_key = jran.key(0)
    satpop_key, mah_key = jran.split(ran_key, 2)

    z_min = 0.4
    z_max = 0.5
    logmp_min = 11.0
    logmp_max = 14.0
    lgmp_min = 10.0
    n_cens = 2_000

    subhalo_model_key = mclsh.DEFAULT_DIFFMAHNET_SAT_MODEL

    # use a mock host halo population for this test
    cenpop = generate_mock_cenpop(z_min, z_max, logmp_min, logmp_max, n_cens=n_cens)

    mc_lg_mu_shmf, _, _, n_mu_per_host = mc_subs.generate_subhalopop(
        satpop_key,
        cenpop.logmp_obs,
        lgmp_min,
    )

    logmsub_obs_shmf = mc_lg_mu_shmf + jnp.repeat(cenpop.logmp_obs, n_mu_per_host)
    t_obs = jnp.repeat(cenpop.t_obs, n_mu_per_host)

    # get the rescaled MAH parameters and MAH's for all halos
    logmp_obs_clipped = jnp.clip(
        logmsub_obs_shmf,
        mclsh.DEFAULT_LOGMSUB_CUTOFF,
        mclsh.DEFAULT_LOGMSUB_HIMASS_CUTOFF,
    )
    mah_params, logmsub_obs_rescaled = apply_mah_rescaling(
        mah_key,
        logmsub_obs_shmf,
        logmp_obs_clipped,
        t_obs,
        cenpop.logt0,
        subhalo_model_key,
    )
    mc_lg_mu_rescaled = logmsub_obs_rescaled - jnp.repeat(
        cenpop.logmp_obs, n_mu_per_host
    )

    assert np.allclose(logmsub_obs_rescaled, logmsub_obs_shmf, rtol=1e-5)
    assert np.allclose(mc_lg_mu_rescaled, mc_lg_mu_shmf, rtol=1e-5)


def test_mc_weighted_lc_subhalos_behaves_as_expected():
    ran_key = jran.key(0)

    z_min = 0.4
    z_max = 0.5
    logmp_min = 11.0
    logmp_max = 14.0
    lgmp_min = 10.0
    n_cens = 2_000

    n_sub_per_host = mclsh.N_LGMU_PER_HOST

    # use a mock host halo population for this test
    cenpop = generate_mock_cenpop(z_min, z_max, logmp_min, logmp_max, n_cens=n_cens)
    halopop = mclsh.weighted_lc_subhalos(cenpop, ran_key, lgmp_min)

    weighted_lc_subhalo_fields = (
        "logmp_obs",
        "t_obs",
        "logt0",
        "nsubhalos",
        "host_index_for_subs",
        "mah_params_subs",
        "logmu_obs",
    )

    for _field in weighted_lc_subhalo_fields:
        assert _field in halopop._fields
        assert np.all(np.isfinite(halopop._asdict()[_field]))

    assert halopop.nsubhalos.shape == (n_cens * n_sub_per_host,)
    assert halopop.logmu_obs.shape == (n_cens * n_sub_per_host,)

    assert halopop.host_index_for_subs.shape == (n_cens * n_sub_per_host,)
    assert halopop.host_index_for_subs.dtype == np.int64
    assert halopop.host_index_for_subs[0] == 0
    assert halopop.host_index_for_subs[-1] == n_cens - 1

    for _key in halopop.mah_params_subs._fields:
        _params = halopop.mah_params_subs._asdict()[_key]
        assert np.all(np.isfinite(_params))
        assert _params.size == n_cens * n_sub_per_host


def test_mc_weighted_lc_subhalos_with_different_nsubs_per_host():
    ran_key = jran.key(0)

    z_min = 0.4
    z_max = 0.5
    logmp_min = 11.0
    logmp_max = 14.0
    lgmp_min = 10.0
    n_cens = 2_000

    n_sub_per_host = mclsh.N_LGMU_PER_HOST + 2

    # use a mock host halo population for this test
    cenpop = generate_mock_cenpop(z_min, z_max, logmp_min, logmp_max, n_cens=n_cens)
    halopop = mclsh.weighted_lc_subhalos(
        cenpop, ran_key, lgmp_min, n_mu_per_host=n_sub_per_host
    )

    weighted_lc_subhalo_fields = (
        "logmp_obs",
        "t_obs",
        "logt0",
        "nsubhalos",
        "host_index_for_subs",
        "mah_params_subs",
        "logmu_obs",
    )

    for _field in weighted_lc_subhalo_fields:
        assert _field in halopop._fields
        assert np.all(np.isfinite(halopop._asdict()[_field]))

    assert halopop.nsubhalos.shape == (n_cens * n_sub_per_host,)
    assert halopop.logmu_obs.shape == (n_cens * n_sub_per_host,)

    assert halopop.host_index_for_subs.shape == (n_cens * n_sub_per_host,)
    assert halopop.host_index_for_subs.dtype == np.int64
    assert halopop.host_index_for_subs[0] == 0
    assert halopop.host_index_for_subs[-1] == n_cens - 1

    for _key in halopop.mah_params_subs._fields:
        _params = halopop.mah_params_subs._asdict()[_key]
        assert np.all(np.isfinite(_params))
        assert _params.size == n_cens * n_sub_per_host


def test_mc_weighted_lc_subhalos_agrees_with_mc_subhalopop():

    ran_key = jran.key(0)

    z_min = 0.4
    z_max = 0.5
    logmp_min = 11.0
    logmp_max = 14.0
    lgmp_min = 10.0
    n_cens = 2_000

    # use a mock host halo population for this test
    cenpop = generate_mock_cenpop(z_min, z_max, logmp_min, logmp_max, n_cens=n_cens)

    n_tests = 5
    lgmp_min_arr = np.linspace(9.0, 10.0, n_tests)
    for lgmp_min in lgmp_min_arr:
        halopop = mclsh.weighted_lc_subhalos(cenpop, ran_key, lgmp_min)

        mc_lg_mu_pop = mc_subs.generate_subhalopop(
            ran_key,
            cenpop.logmp_obs,
            lgmp_min,
        )[0]

        assert np.allclose(
            mc_lg_mu_pop.size,
            halopop.nsubhalos.sum(),
            rtol=0.1,
        )
