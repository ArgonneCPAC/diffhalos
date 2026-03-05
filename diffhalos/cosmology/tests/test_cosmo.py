""" """

import numpy as np

from ..cosmo import (
    get_tobs_from_zobs,
    DEFAULT_COSMO_PRIORS,
    DEFAULT_COSMOLOGY,
    DEFAULT_COSMO_NAMES,
    N_DEFAULT_COSMO_PARAMS,
    DEFAULT_COSMOLOGY_ARRAY,
    DEFAULT_COSMOLOGY_COLOSSUS,
    DEFAULT_COSMOLOGY_DSPS,
)


def test_get_tobs_from_zobs_evaluates():

    z_obs = np.linspace(0.1, 2.0, 100)

    t_obs, t_0 = get_tobs_from_zobs(z_obs)

    assert np.all(np.isfinite(t_obs))
    assert np.isfinite(t_0)
    assert t_obs[-1] <= t_0


def test_cosmo_params():

    assert isinstance(DEFAULT_COSMO_PRIORS, dict)
    assert isinstance(DEFAULT_COSMOLOGY, dict)
    assert isinstance(DEFAULT_COSMOLOGY_COLOSSUS, dict)
    assert isinstance(DEFAULT_COSMOLOGY_DSPS, tuple)
    assert (
        len(DEFAULT_COSMO_NAMES)
        == len(DEFAULT_COSMOLOGY_ARRAY)
        == N_DEFAULT_COSMO_PARAMS
    )

    for i, _param in enumerate(DEFAULT_COSMO_NAMES):
        assert DEFAULT_COSMOLOGY[_param] == DEFAULT_COSMOLOGY_ARRAY[i]

    assert np.all(
        [_p in DEFAULT_COSMOLOGY.keys() for _p in DEFAULT_COSMO_PRIORS.keys()]
    )
    assert np.all(np.isfinite(DEFAULT_COSMOLOGY_DSPS))
