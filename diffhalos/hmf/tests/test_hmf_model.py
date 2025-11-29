""" """

import numpy as np
from diffsky.experimental import mc_lightcone_halos as mclh
from diffsky.mass_functions import mc_hosts as diffsky_mc_hosts
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from .. import hmf_model


def test_cuml_hmf_evaluations():
    lgmp_arr = np.linspace(-6, 0, 500)
    redshift = 0.2
    res = hmf_model.predict_cuml_hmf(hmf_model.DEFAULT_HMF_PARAMS, lgmp_arr, redshift)
    assert res.shape == lgmp_arr.shape
    assert np.all(np.isfinite(res))


def test_diff_hmf_evaluations():
    lgmp_arr = np.linspace(-6, 0, 500)
    redshift = 0.2
    res = hmf_model.predict_differential_hmf(
        hmf_model.DEFAULT_HMF_PARAMS, lgmp_arr, redshift
    )
    assert res.shape == lgmp_arr.shape
    assert np.all(np.isfinite(res))


def test_halo_lightcone_weights_against_diffsky():
    lgmp_min, lgmp_max = 11, 17
    z_min, z_max = 0.02, 3.0
    n_per_dim = 500
    lgmp_grid = np.linspace(lgmp_min, lgmp_max, n_per_dim)
    z_grid = np.linspace(z_min, z_max, n_per_dim)

    sky_area_degsq = 200.0

    ran_key = jran.key(0)

    cenpop = mclh.get_weighted_lightcone_grid_host_halo_diffmah(
        ran_key, lgmp_grid, z_grid, sky_area_degsq
    )

    nhalos = hmf_model.halo_lightcone_weights(
        cenpop["logmp_obs"],
        cenpop["z_obs"],
        sky_area_degsq,
        hmf_params=diffsky_mc_hosts.DEFAULT_HMF_PARAMS,
        cosmo_params=DEFAULT_COSMOLOGY,
    )

    assert np.allclose(nhalos, cenpop["nhalos"], rtol=0.01)
