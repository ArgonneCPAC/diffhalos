""" """

import numpy as np

from ..cosmo_basics import get_tobs_from_zobs


def test_get_tobs_from_zobs_evaluates():

    z_obs = np.linspace(0.1, 2.0, 100)

    t_obs, t_0 = get_tobs_from_zobs(z_obs)

    assert np.all(np.isfinite(t_obs))
    assert np.isfinite(t_0)
    assert t_obs[-1] <= t_0
