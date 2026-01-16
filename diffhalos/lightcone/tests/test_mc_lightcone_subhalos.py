""" """

import numpy as np
from jax import random as jran
from collections import namedtuple

from .. import mc_lightcone_halos as mclh
from .. import mc_lightcone_subhalos as mclsh
from ...ccshmf import mc_subs


def test_mc_lightcone_subhalo_mass_function_works_as_expected():
    ran_key = jran.key(0)
    uran_key, counts_key = jran.split(ran_key, 2)

    nhost = 10
    lgmhost = np.linspace(13.0, 15.0, nhost)
    lgmp_min = 12.0

    subhalo_counts_per_halo, nsubs_tot = mc_subs.get_mean_subhalo_counts_poisson(
        counts_key,
        lgmhost,
        lgmp_min,
    )

    (
        mc_lg_mu,
        lgmhost_pop,
        host_halo_indx,
    ) = mclsh.mc_lightcone_subhalo_mass_function(
        uran_key,
        lgmhost,
        lgmp_min,
        subhalo_counts_per_halo,
        int(nsubs_tot),
    )

    assert mc_lg_mu.size == lgmhost_pop.size == host_halo_indx.size == nsubs_tot
    assert np.all(np.isfinite(mc_lg_mu))
    assert np.all(np.isfinite(lgmhost_pop))
    assert np.all(np.isfinite(host_halo_indx))


def test_mc_weighted_subhalo_lightcone_behaves_as_expected():
    ran_key = jran.key(0)

    n_halos = 500
    z_obs = np.linspace(0.45, 0.5, n_halos)
    lgmp_max = 15.0
    sky_area_degsq = 1.0
    logt0 = np.log10(13.8)
    n_sub_per_host = mclsh.N_LGMU_PER_HOST

    n_tests = 5
    lgmp_min_arr = np.linspace(11.0, 13.0, n_tests)
    for lgmp_min in lgmp_min_arr:
        test_key, ran_key = jran.split(ran_key, 2)

        logmp_obs = np.linspace(lgmp_min, lgmp_max, n_halos)

        cenpop = mclh.weighted_lightcone_host_halo(
            test_key,
            z_obs,
            logmp_obs,
            sky_area_degsq,
        )

        satpop = mclsh.mc_weighted_subhalo_lightcone(
            cenpop,
            test_key,
            lgmp_min,
            logt0,
        )

        assert "nsubhalos" in satpop._fields
        assert "logmu_subs" in satpop._fields
        assert "host_index_for_sub" in satpop._fields
        assert "mah_params_sub" in satpop._fields

        assert np.all(np.isfinite(satpop.nsubhalos))
        assert satpop.nsubhalos.shape == (n_halos * n_sub_per_host,)

        assert np.all(np.isfinite(satpop.logmu_subs))
        assert satpop.logmu_subs.shape == (n_halos * n_sub_per_host,)

        assert satpop.host_index_for_sub.shape == (n_halos * n_sub_per_host,)
        assert satpop.host_index_for_sub.dtype == np.int64
        assert satpop.host_index_for_sub[0] == 0
        assert satpop.host_index_for_sub[-1] == n_halos - 1

        assert np.all(np.isfinite(satpop.mah_params_sub))
        for _key in satpop.mah_params_sub._fields:
            _params = satpop.mah_params_sub._asdict()[_key]
            assert np.all(np.isfinite(_params))
            assert _params.size == n_halos * n_sub_per_host


def test_mc_weighted_subhalo_lightcone_with_different_nsubs_per_host():
    ran_key = jran.key(0)

    n_halos = 500
    z_obs = np.linspace(0.45, 0.5, n_halos)
    lgmp_max = 15.0
    sky_area_degsq = 1.0
    logt0 = np.log10(13.8)
    n_sub_per_host = 10

    n_tests = 5
    lgmp_min_arr = np.linspace(11.0, 13.0, n_tests)
    for lgmp_min in lgmp_min_arr:
        test_key, ran_key = jran.split(ran_key, 2)

        logmp_obs = np.linspace(lgmp_min, lgmp_max, n_halos)

        cenpop = mclh.weighted_lightcone_host_halo(
            test_key,
            z_obs,
            logmp_obs,
            sky_area_degsq,
        )

        satpop = mclsh.mc_weighted_subhalo_lightcone(
            cenpop,
            test_key,
            lgmp_min,
            logt0,
            n_mu_per_host=n_sub_per_host,
        )

        assert "nsubhalos" in satpop._fields
        assert "logmu_subs" in satpop._fields
        assert "host_index_for_sub" in satpop._fields
        assert "mah_params_sub" in satpop._fields

        assert np.all(np.isfinite(satpop.nsubhalos))
        assert satpop.nsubhalos.shape == (n_halos * n_sub_per_host,)

        assert np.all(np.isfinite(satpop.logmu_subs))
        assert satpop.logmu_subs.shape == (n_halos * n_sub_per_host,)

        assert satpop.host_index_for_sub.shape == (n_halos * n_sub_per_host,)
        assert satpop.host_index_for_sub.dtype == np.int64
        assert satpop.host_index_for_sub[0] == 0
        assert satpop.host_index_for_sub[-1] == n_halos - 1

        assert np.all(np.isfinite(satpop.mah_params_sub))
        for _key in satpop.mah_params_sub._fields:
            _params = satpop.mah_params_sub._asdict()[_key]
            assert np.all(np.isfinite(_params))
            assert _params.size == n_halos * n_sub_per_host


def test_mc_weighted_subhalo_lightcone_agrees_with_mc_subhalopop():

    ran_key = jran.key(0)
    uran_key, counts_key = jran.split(ran_key, 2)

    logt0 = np.log10(13.8)
    n_halos = 500

    lgmhost_arr = np.linspace(11.0, 13.0, n_halos)
    t_obs_arr = np.ones(n_halos) * 13.5
    cenpop_dict = {"logmp_obs": lgmhost_arr, "t_obs": t_obs_arr}
    cenpop = namedtuple("halopop", cenpop_dict.keys())(**cenpop_dict)

    n_tests = 5
    lgmp_min_arr = np.linspace(9.0, 10.0, n_tests)
    for lgmp_min in lgmp_min_arr:
        satpop = mclsh.mc_weighted_subhalo_lightcone(
            cenpop,
            uran_key,
            lgmp_min,
            logt0,
        )

        subhalo_counts_per_halo, nsubs_tot = mc_subs.get_mean_subhalo_counts_poisson(
            counts_key,
            lgmhost_arr,
            lgmp_min,
        )

        mc_lg_mu = mc_subs.generate_subhalopop(
            ran_key,
            lgmhost_arr,
            lgmp_min,
            subhalo_counts_per_halo,
            int(subhalo_counts_per_halo.sum()),
        )[0]

        assert np.allclose(
            mc_lg_mu.size,
            satpop.nsubhalos.sum(),
            rtol=0.1,
        )
