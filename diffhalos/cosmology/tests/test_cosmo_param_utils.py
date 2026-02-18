""""""

import numpy as np

from ..cosmo_params import sample_cosmo_params, DEFAULT_COSMO_PRIORS


def test_sample_cosmo_params():
    cosmo_params = DEFAULT_COSMO_PRIORS.keys()
    n_samples = 10
    n_params = len(cosmo_params)

    cosmo_param_samples = sample_cosmo_params(
        cosmo_priors=DEFAULT_COSMO_PRIORS,
        seed=None,
        num_samples=n_samples,
    )

    assert cosmo_param_samples.shape == (n_samples, n_params)
    for i in range(n_samples):
        assert np.all(np.isfinite(cosmo_param_samples[i]))
        for j, param in enumerate(cosmo_params):
            assert cosmo_param_samples[i][j] >= DEFAULT_COSMO_PRIORS[param][0]
            assert cosmo_param_samples[i][j] <= DEFAULT_COSMO_PRIORS[param][1]
