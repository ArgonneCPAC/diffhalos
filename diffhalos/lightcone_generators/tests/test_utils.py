""" """

import numpy as np

from ..utils import generate_mock_cenpop


def test_generate_mock_cenpop_behaves_as_expected():

    z_min = 0.2
    z_max = 2.0
    logmp_min = 11.0
    logmp_max = 14.0
    n_cens = 100

    cenpop = generate_mock_cenpop(
        z_min,
        z_max,
        logmp_min,
        logmp_max,
        n_cens=n_cens,
    )

    assert "t_obs" in cenpop._fields
    assert "logmp_obs" in cenpop._fields
    assert "logt0" in cenpop._fields

    assert np.all(np.isfinite(cenpop.t_obs))
    assert np.all(np.isfinite(cenpop.logmp_obs))
    assert np.isfinite(cenpop.logt0)
    assert np.all(cenpop.t_obs <= 10**cenpop.logt0)
    assert np.all((cenpop.logmp_obs >= logmp_min) * (cenpop.logmp_obs <= logmp_max))
