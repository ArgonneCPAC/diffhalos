""" """

import numpy as np
from jax import random as jran

from .. import mc_lightcone_halos as mclh
from .. import mc_lightcone_subhalos as mclsh
from ...calibrations.ccshmf_cal import DEFAULT_CCSHMF_PARAMS  # noqa


def test_mc_weighted_halo_lightcone():
    """
    Enforce mc_lightcone_host_halo_diffmah
    returns reasonable results when passed
    alternative halo mass function parameters
    """

    ran_key = jran.key(0)
    z_max = 0.5
    z_min = z_max - 0.05
    lgmp_max = 15.0
    sky_area_degsq = 1.0
    num_halos = 500

    n_tests = 5
    lgmp_min_arr = np.linspace(11.0, 13.0, n_tests)
    for lgmp_min in lgmp_min_arr:
        test_key, ran_key = jran.split(ran_key, 2)

        cenpop = mclh.mc_weighted_halo_lightcone(
            test_key,
            num_halos,
            z_min,
            z_max,
            lgmp_min,
            lgmp_max,
            sky_area_degsq,
        )

        satpop = mclsh.mc_weighted_subhalo_lightcone(
            cenpop,
            lgmp_min,
            ccshmf_params=DEFAULT_CCSHMF_PARAMS,
        )

        n_gals = satpop["z_obs"].size
        assert satpop["logmp_obs"].size == satpop["logmp0"].size == n_gals == num_halos
        assert np.all(np.isfinite(satpop["z_obs"]))

        assert np.all(satpop["z_obs"] >= z_min)
        assert np.all(satpop["z_obs"] <= z_max)

        assert np.all(satpop["logmp_obs"] >= lgmp_min)
        assert np.all(satpop["logmp_obs"] <= lgmp_max)

        assert np.all(np.isfinite(satpop["nsubhalos"]))
        assert satpop["nsubhalos"].shape == (
            satpop["logmp_obs"].size,
            mclsh.N_LGMU_TABLE,
        )

        assert np.all(np.isfinite(satpop["logmu_subs"]))
        assert satpop["logmu_subs"].shape == (
            satpop["logmp_obs"].size,
            mclsh.N_LGMU_TABLE,
        )
