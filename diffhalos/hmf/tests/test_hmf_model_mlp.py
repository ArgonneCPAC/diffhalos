""" """

import numpy as np

from jax import random as jran

from diffsky.experimental import mc_lightcone_halos as mclh

from .. import hmf_model_mlp
from ...cosmology.cosmo import DEFAULT_COSMOLOGY_ARRAY
from ...cosmology.cosmo_param_utils import define_dsps_cosmology
from ...cosmology.geometry_utils import compute_volume_from_sky_area


def test_cuml_hmf_evaluations():
    lgmp_arr = np.linspace(-6, 0, 500)
    redshift = 0.2
    res = hmf_model_mlp.predict_cuml_hmf(
        DEFAULT_COSMOLOGY_ARRAY,
        lgmp_arr,
        redshift,
    )
    assert res.shape == lgmp_arr.shape
    assert np.all(np.isfinite(res))


def test_diff_hmf_evaluations():
    lgmp_arr = np.linspace(-6, 0, 500)
    redshift = 0.2
    res = hmf_model_mlp.predict_diff_hmf(
        DEFAULT_COSMOLOGY_ARRAY,
        lgmp_arr,
        redshift,
    )
    assert res.shape == lgmp_arr.shape
    assert np.all(np.isfinite(res))


def test_halo_lightcone_weights():
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

    nhalos = hmf_model_mlp.halo_lightcone_weights(
        cenpop["logmp_obs"],
        cenpop["z_obs"],
        sky_area_degsq,
        cosmo_params=DEFAULT_COSMOLOGY_ARRAY,
    )

    assert np.all(np.isfinite(nhalos))
    assert nhalos.size == cenpop["logmp_obs"].size


def test_get_mean_nhalos_from_volume_and_from_sky_area():

    lgmp_min = 11.0
    lgmp_max = 13.0

    sky_area_degsq = 10.0

    redshift = np.linspace(0.1, 3.0, 10)

    volume_com_mpc = compute_volume_from_sky_area(
        redshift,
        sky_area_degsq,
        define_dsps_cosmology(DEFAULT_COSMOLOGY_ARRAY),
    )

    nhalos_from_vol = hmf_model_mlp.get_mean_nhalos_from_volume(
        redshift,
        volume_com_mpc,
        DEFAULT_COSMOLOGY_ARRAY,
        lgmp_min,
        lgmp_max,
    )

    nhalos_from_area = hmf_model_mlp.get_mean_nhalos_from_sky_area(
        redshift,
        sky_area_degsq,
        DEFAULT_COSMOLOGY_ARRAY,
        lgmp_min,
        lgmp_max,
    )

    assert np.all(np.isfinite(nhalos_from_vol))
    assert np.all(np.isfinite(nhalos_from_area))
    assert np.allclose(nhalos_from_vol, nhalos_from_area)
