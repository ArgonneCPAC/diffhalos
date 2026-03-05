""""""

import numpy as np

from ..cosmo_param_utils import (
    sample_cosmo_params,
    define_dsps_cosmology,
    DEFAULT_COSMO_PRIORS,
    DEFAULT_COSMOLOGY,
)
from ..cosmo import DEFAULT_COSMOLOGY_ARRAY


def test_sample_cosmo_params():
    cosmo_params = DEFAULT_COSMOLOGY.keys()
    cosmo_sampled = DEFAULT_COSMO_PRIORS.keys()

    n_samples = 10
    n_params = len(cosmo_params)

    cosmo_param_samples = sample_cosmo_params(
        underlying_cosmo=DEFAULT_COSMOLOGY,
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

    dsps_cosmo = define_dsps_cosmology(DEFAULT_COSMOLOGY_ARRAY)

    DEFAULT_COSMOLOGY["h"] = DEFAULT_COSMOLOGY["H0"] / 100
    assert isinstance(dsps_cosmo, tuple)
    for _param in dsps_cosmo._fields:
        assert getattr(dsps_cosmo, _param) == DEFAULT_COSMOLOGY[_param]
