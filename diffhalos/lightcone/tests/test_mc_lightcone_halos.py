""" """

import numpy as np
from jax import random as jran

from dsps.cosmology import flat_wcdm, DEFAULT_COSMOLOGY

from ...hmf import mc_hosts
from ...calibrations.hmf_cal import hacc_core_hmf_params as hchmf
from .. import mc_lightcone_halos as mclh


def test_mc_lightcone_host_halo_mass_function():
    """
    Enforce mc_lightcone_host_halo_mass_function
    produces halo mass functions that are consistent
    diffsky.mass_functions.mc_hosts evaluated at the median redshift
    """
    lgmp_min = 12.0
    z_min, z_max = 0.4, 0.5
    n_grid_z = 500
    z_grid = np.linspace(z_min, z_max, n_grid_z)
    sky_area_degsq = 10.0

    n_tests = 5
    ran_keys = jran.split(jran.key(0), n_tests)
    for ran_key in ran_keys:
        args = (ran_key, lgmp_min, z_grid, sky_area_degsq)

        (
            redshifts_galpop,
            logmp_halopop,
        ) = mclh.mc_lightcone_host_halo_mass_function(*args)

        assert np.all(np.isfinite(redshifts_galpop))
        assert np.all(np.isfinite(logmp_halopop))
        assert logmp_halopop.shape == redshifts_galpop.shape
        assert np.all(redshifts_galpop >= z_min)
        assert np.all(redshifts_galpop <= z_max)
        assert np.all(logmp_halopop > lgmp_min)

        z_med = np.median(redshifts_galpop)

        cosmo = DEFAULT_COSMOLOGY
        vol_lo = (
            (4 / 3)
            * np.pi
            * flat_wcdm.comoving_distance_to_z(
                z_min,
                *cosmo,
            )
            ** 3
        )
        vol_hi = (
            (4 / 3)
            * np.pi
            * flat_wcdm.comoving_distance_to_z(
                z_max,
                *cosmo,
            )
            ** 3
        )
        fsky = sky_area_degsq / mclh.FULL_SKY_AREA
        vol_com_mpc = fsky * (vol_hi - vol_lo)

        lgmp_halopop_zmed = mc_hosts.mc_host_halos_singlez(
            ran_key, lgmp_min, z_med, vol_com_mpc
        )

        n_lightcone, n_snapshot = redshifts_galpop.size, lgmp_halopop_zmed.size
        fracdiff = (n_lightcone - n_snapshot) / n_snapshot
        assert np.abs(fracdiff) < 0.05

        lgmp_hist_lc, lgmp_bins = np.histogram(logmp_halopop, bins=50)
        lgmp_hist_zmed, lgmp_bins = np.histogram(
            lgmp_halopop_zmed,
            bins=lgmp_bins,
        )
        msk_counts = lgmp_hist_zmed > 500
        fracdiff = (
            lgmp_hist_lc[msk_counts] - lgmp_hist_zmed[msk_counts]
        ) / lgmp_hist_zmed[msk_counts]
        assert np.all(np.abs(fracdiff) < 0.1)


def test_mc_lightcone_host_halo_mass_function_lgmp_max_feature():
    """
    Using lgmp_max gives the same halo counts
    as when not using it and masking
    """

    ran_key = jran.key(0)
    lgmp_min = 12.0
    z_min, z_max = 0.4, 0.5
    n_grid_z = 500
    z_grid = np.linspace(z_min, z_max, n_grid_z)
    sky_area_degsq = 200.0

    lgmp_max_test = 13.0

    n_tests = 10
    for __ in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)
        args = (test_key, lgmp_min, z_grid, sky_area_degsq)

        z_halopop, logmp_halopop = mclh.mc_lightcone_host_halo_mass_function(
            *args,
        )
        z_halopop2, logmp_halopop2 = mclh.mc_lightcone_host_halo_mass_function(
            *args, lgmp_max=lgmp_max_test
        )
        assert z_halopop.size > z_halopop2.size
        assert z_halopop2.size == logmp_halopop2.size

        n1 = np.sum(logmp_halopop < lgmp_max_test)
        n2 = logmp_halopop2.size
        assert np.allclose(n1, n2, rtol=0.02)


def test_nhalo_weighted_lc_grid():
    """
    Should get within 10% of Nhalos of a Monte Carlo generated lightcone
    """

    ran_key = jran.key(0)
    n_tests = 10

    sky_area_degsq = 15.0

    for itest in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)

        lgm_key, z_key, lc_key = jran.split(test_key, 3)
        lgmp_min = jran.uniform(lgm_key, minval=11, maxval=12, shape=())
        z_min = jran.uniform(z_key, minval=0.1, maxval=1.0, shape=())
        z_max = z_min + 1

        n_z, n_m = 150, 250
        lgmp_grid = np.linspace(lgmp_min, 15, n_m)
        z_grid = np.linspace(z_min, z_max, n_z)

        nhalo_weighted_lc_grid = mclh.get_nhalo_weighted_lc_grid(
            lgmp_grid,
            z_grid,
            sky_area_degsq,
            hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
            cosmo_params=DEFAULT_COSMOLOGY,
        )
        assert nhalo_weighted_lc_grid.shape == (n_z, n_m)

        args = (lc_key, lgmp_min, z_grid, sky_area_degsq)
        z_halopop, logmp_halopop = mclh.mc_lightcone_host_halo_mass_function(
            *args,
        )

        assert np.allclose(
            logmp_halopop.size,
            nhalo_weighted_lc_grid.sum(),
            rtol=0.1,
        )


def test_mc_weighted_halo_lightcone_sobol():
    ran_key = jran.key(0)
    n_tests = 10

    sky_area_degsq = 15.0

    num_halos = 1_000

    for itest in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)

        lgm_key, z_key, lc_key = jran.split(test_key, 3)
        lgmp_min = jran.uniform(lgm_key, minval=11, maxval=12, shape=())
        lgmp_max = jran.uniform(lgm_key, minval=13, maxval=14, shape=())
        z_min = jran.uniform(z_key, minval=0.1, maxval=1.0, shape=())
        z_max = z_min + 1

        cenpop = mclh.mc_weighted_halo_lightcone(
            ran_key,
            num_halos,
            z_min,
            z_max,
            lgmp_min,
            lgmp_max,
            sky_area_degsq,
            grid_scheme="sobol",
        )

        assert np.all(np.isfinite(cenpop["nhalos"]))
        assert cenpop["logmp_obs"].size == num_halos
        assert np.sum(cenpop["mp_is_sub"]) == 0

        assert np.all(cenpop["z_obs"] >= z_min)
        assert np.all(cenpop["z_obs"] <= z_max)

        # some halos with logmp_obs<lgmp_min or logmp_obs>lgmp_max is ok,
        # but too many indicates an issue with diffmahnet replicating logmp_obs
        assert np.mean(cenpop["logmp_obs"] < lgmp_min) < 0.2
        assert np.mean(cenpop["logmp_obs"] > lgmp_max) < 0.2


def test_mc_weighted_halo_lightcone_stratified():
    ran_key = jran.key(0)
    n_tests = 10

    sky_area_degsq = 15.0

    n_per_dim = 10
    num_halos = n_per_dim**2

    for itest in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)

        lgm_key, z_key, lc_key = jran.split(test_key, 3)
        lgmp_min = jran.uniform(lgm_key, minval=11, maxval=12, shape=())
        lgmp_max = jran.uniform(lgm_key, minval=13, maxval=14, shape=())
        z_min = jran.uniform(z_key, minval=0.1, maxval=1.0, shape=())
        z_max = z_min + 1

        cenpop = mclh.mc_weighted_halo_lightcone(
            ran_key,
            num_halos,
            z_min,
            z_max,
            lgmp_min,
            lgmp_max,
            sky_area_degsq,
            grid_scheme="stratified",
            n_per_dim=n_per_dim,
        )

        assert np.all(np.isfinite(cenpop["nhalos"]))
        assert cenpop["logmp_obs"].size == num_halos
        assert np.sum(cenpop["mp_is_sub"]) == 0

        assert np.all(cenpop["z_obs"] >= z_min)
        assert np.all(cenpop["z_obs"] <= z_max)

        # some halos with logmp_obs<lgmp_min or logmp_obs>lgmp_max is ok,
        # but too many indicates an issue with diffmahnet replicating logmp_obs
        assert np.mean(cenpop["logmp_obs"] < lgmp_min) < 0.2
        assert np.mean(cenpop["logmp_obs"] > lgmp_max) < 0.2


def test_mc_weighted_halo_lightcone_input_grid():
    ran_key = jran.key(0)
    n_tests = 10

    sky_area_degsq = 15.0

    num_halos = 1_000
    for itest in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)

        lgm_key, z_key, lc_key = jran.split(test_key, 3)
        lgmp_min = jran.uniform(lgm_key, minval=11, maxval=12, shape=())
        lgmp_max = jran.uniform(lgm_key, minval=13, maxval=14, shape=())
        z_min = jran.uniform(z_key, minval=0.1, maxval=1.0, shape=())
        z_max = z_min + 1

        z_grid = np.linspace(z_min, z_max, num_halos)
        lgmp_grid = np.linspace(lgmp_min, lgmp_max, num_halos)

        cenpop = mclh.mc_weighted_halo_lightcone(
            ran_key,
            num_halos,
            z_min,
            z_max,
            lgmp_min,
            lgmp_max,
            sky_area_degsq,
            grid_scheme="input_grid",
            z_obs=z_grid,
            logmp_obs=lgmp_grid,
        )

        assert np.all(np.isfinite(cenpop["nhalos"]))
        assert cenpop["logmp_obs"].size == num_halos
        assert np.sum(cenpop["mp_is_sub"]) == 0

        assert np.all(cenpop["z_obs"] >= z_min)
        assert np.all(cenpop["z_obs"] <= z_max)

        # some halos with logmp_obs<lgmp_min or logmp_obs>lgmp_max is ok,
        # but too many indicates an issue with diffmahnet replicating logmp_obs
        assert np.mean(cenpop["logmp_obs"] < lgmp_min) < 0.2
        assert np.mean(cenpop["logmp_obs"] > lgmp_max) < 0.2


def test_mc_lightcone_host_halo_diffmah():
    """
    Enforce mc_lightcone_host_halo_diffmah returns reasonable results
    """

    ran_key = jran.key(0)
    lgmp_min = 12.0
    sky_area_degsq = 1.0
    n_grid_z = 500

    n_tests = 5
    z_max_arr = np.linspace(0.2, 2.5, n_tests)
    for z_max in z_max_arr:
        test_key, ran_key = jran.split(ran_key, 2)
        z_min = z_max - 0.05
        z_grid = np.linspace(z_min, z_max, n_grid_z)
        args = (test_key, lgmp_min, z_grid, sky_area_degsq)

        cenpop = mclh.mc_lightcone_host_halo_diffmah(*args)
        n_gals = cenpop["z_obs"].size
        assert cenpop["logmp_obs"].size == cenpop["logmp0"].size == n_gals
        assert np.all(np.isfinite(cenpop["z_obs"]))

        assert np.all(cenpop["z_obs"] >= z_min)
        assert np.all(cenpop["z_obs"] <= z_max)

        assert np.sum(cenpop["mp_is_sub"] == 0)

        # some halos with logmp_obs<lgmp_min is ok,
        # but too many indicates an issue with diffmahnet replicating logmp_obs
        assert np.mean(cenpop["logmp_obs"] < lgmp_min) < 0.2, f"z_min={z_min:.2f}"


def test_get_weighted_lightcone_grid_host_halo_diffmah():
    ran_key = jran.key(0)
    n_tests = 10

    sky_area_degsq = 15.0

    lgmp_max = 15

    for itest in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)

        lgm_key, z_key, lc_key = jran.split(test_key, 3)
        lgmp_min = jran.uniform(lgm_key, minval=11, maxval=12, shape=())
        z_min = jran.uniform(z_key, minval=0.1, maxval=1.0, shape=())
        z_max = z_min + 1

        n_z, n_m = 150, 250
        lgmp_grid = np.linspace(lgmp_min, lgmp_max, n_m)
        z_grid = np.linspace(z_min, z_max, n_z)

        cenpop = mclh.get_weighted_lightcone_grid_host_halo_diffmah(
            ran_key, lgmp_grid, z_grid, sky_area_degsq
        )
        assert np.all(np.isfinite(cenpop["nhalos"]))
        assert cenpop["logmp_obs"].size == n_z * n_m
        assert np.sum(cenpop["mp_is_sub"]) == 0

        assert np.all(cenpop["z_obs"] >= z_min)
        assert np.all(cenpop["z_obs"] <= z_max)

        # some halos with logmp_obs<lgmp_min or logmp_obs>lgmp_max is ok,
        # but too many indicates an issue with diffmahnet replicating logmp_obs
        assert np.mean(cenpop["logmp_obs"] < lgmp_min) < 0.2
        assert np.mean(cenpop["logmp_obs"] > lgmp_max) < 0.2


def test_mc_lightcone_host_halo_diffmah_alt_mf_params():
    """
    Enforce mc_lightcone_host_halo_diffmah returns
    reasonable results when passed
    alternative halo mass function parameters
    """
    ran_key = jran.key(0)
    lgmp_min = 12.0
    sky_area_degsq = 1.0
    n_grid_z = 500

    n_tests = 5
    z_max_arr = np.linspace(0.2, 2.5, n_tests)
    for z_max in z_max_arr:
        test_key, ran_key = jran.split(ran_key, 2)
        z_min = z_max - 0.05
        z_grid = np.linspace(z_min, z_max, n_grid_z)
        args = (test_key, lgmp_min, z_grid, sky_area_degsq)

        cenpop = mclh.mc_lightcone_host_halo_diffmah(
            *args,
            hmf_params=hchmf.HMF_PARAMS,
        )
        n_gals = cenpop["z_obs"].size
        assert cenpop["logmp_obs"].size == cenpop["logmp0"].size == n_gals
        assert np.all(np.isfinite(cenpop["z_obs"]))

        assert np.all(cenpop["z_obs"] >= z_min)
        assert np.all(cenpop["z_obs"] <= z_max)

        assert np.sum(cenpop["mp_is_sub"] == 0)

        # Some halos with logmp_obs<lgmp_min is ok,
        # but too many indicates an issue with diffmahnet replicating logmp_obs
        assert np.mean(cenpop["logmp_obs"] < lgmp_min) < 0.2, f"z_min={z_min:.2f}"


def test_mc_weighted_halo_lightcone():
    """
    Enforce mc_weighted_halo_lightcone
    returns reasonable results asked
    for the default behavior
    """

    ran_key = jran.key(0)
    lgmp_min = 12.0
    lgmp_max = 15.0
    sky_area_degsq = 1.0
    num_halos = 500

    n_tests = 5
    z_max_arr = np.linspace(0.2, 2.5, n_tests)
    for z_max in z_max_arr:
        test_key, ran_key = jran.split(ran_key, 2)
        z_min = z_max - 0.05

        cenpop = mclh.mc_weighted_halo_lightcone(
            test_key,
            num_halos,
            z_min,
            z_max,
            lgmp_min,
            lgmp_max,
            sky_area_degsq,
        )
        n_gals = cenpop["z_obs"].size
        assert cenpop["logmp_obs"].size == cenpop["logmp0"].size == n_gals == num_halos
        assert np.all(np.isfinite(cenpop["z_obs"]))

        assert np.all(cenpop["z_obs"] >= z_min)
        assert np.all(cenpop["z_obs"] <= z_max)

        # some halos with logmp_obs<lgmp_min or logmp_obs>lgmp_max is ok,
        # but too many indicates an issue with diffmahnet replicating logmp_obs
        assert np.mean(cenpop["logmp_obs"] < lgmp_min) < 0.2
        assert np.mean(cenpop["logmp_obs"] > lgmp_max) < 0.2
