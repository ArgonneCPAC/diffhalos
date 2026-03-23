""""""

import numpy as np
from collections import namedtuple

from ..cosmo_param_utils import (
    sample_cosmo_params,
    sample_cosmo_params_full_cosmo,
    define_dsps_cosmology,
    define_colossus_cosmology,
    define_full_cosmology,
    DEFAULT_COSMO_PRIORS,
    DEFAULT_COSMOLOGY_DICT,
    DEFAULT_COSMOLOGY_NTUP,
)


def test_sample_cosmo_params_cosmo():
    cosmo_sampled = DEFAULT_COSMO_PRIORS.keys()

    n_samples = 10
    n_params = len(cosmo_sampled)

    cosmo_param_samples = sample_cosmo_params(
        cosmo_priors=DEFAULT_COSMO_PRIORS,
        seed=1291,
        num_samples=n_samples,
    )

    assert cosmo_param_samples.shape == (n_samples, n_params)

    for i in range(n_samples):
        assert np.all(np.isfinite(cosmo_param_samples[i]))
        for j, param in enumerate(cosmo_sampled):
            assert cosmo_param_samples[i][j] >= DEFAULT_COSMO_PRIORS[param][0]
            assert cosmo_param_samples[i][j] <= DEFAULT_COSMO_PRIORS[param][1]


def test_sample_cosmo_params_full_cosmo():
    cosmo_params = DEFAULT_COSMOLOGY_DICT.keys()
    cosmo_sampled = DEFAULT_COSMO_PRIORS.keys()

    n_samples = 10
    n_params = len(cosmo_params)

    cosmo_param_samples = sample_cosmo_params_full_cosmo(
        underlying_cosmo=DEFAULT_COSMOLOGY_DICT,
        cosmo_priors=DEFAULT_COSMO_PRIORS,
        seed=1291,
        num_samples=n_samples,
    )

    assert cosmo_param_samples.shape == (n_samples, n_params)

    for i in range(n_samples):
        assert np.all(np.isfinite(cosmo_param_samples[i]))
        for j, param in enumerate(cosmo_params):
            if param in cosmo_sampled:
                assert cosmo_param_samples[i][j] >= DEFAULT_COSMO_PRIORS[param][0]
                assert cosmo_param_samples[i][j] <= DEFAULT_COSMO_PRIORS[param][1]


def test_define_dsps_cosmology():

    dsps_cosmo = define_dsps_cosmology(DEFAULT_COSMOLOGY_NTUP)

    assert isinstance(dsps_cosmo, tuple)
    for _param in dsps_cosmo._fields:
        assert getattr(dsps_cosmo, _param) == DEFAULT_COSMOLOGY_DICT[_param]


def test_define_colossus_cosmology():

    Om0 = 0.34
    sigma8 = 0.5
    H0 = 90.0

    colossus_cosmo = define_colossus_cosmology(
        DEFAULT_COSMOLOGY_DICT,
        cosmo_name="ColossusCosmo",
        Om0=Om0,
        sigma8=sigma8,
        H0=H0,
    )

    assert colossus_cosmo.Om0 == Om0
    assert colossus_cosmo.sigma8 == sigma8
    assert colossus_cosmo.H0 == H0


def test_define_full_cosmology():

    Om0 = 0.34
    sigma8 = 0.5
    H0 = 90.0

    cosmo_params_ntup = namedtuple("cosmo", "Om0 sigma8 H0")(*(Om0, sigma8, H0))

    full_cosmo = define_full_cosmology(
        cosmo_params_ntup,
        underlying_cosmo=DEFAULT_COSMOLOGY_NTUP,
    )

    assert full_cosmo.Om0 == Om0
    assert full_cosmo.sigma8 == sigma8
    assert full_cosmo.H0 == H0
