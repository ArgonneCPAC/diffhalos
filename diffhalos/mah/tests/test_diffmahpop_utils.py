""" """

import numpy as np
from jax import numpy as jnp
from jax import random as jran

from ..diffmahpop_utils import mc_mah_cenpop, DEFAULT_DIFFMAHPOP_PARAMS


def test_mc_mah_cenpop_behaves_as_expected():

    ran_key = jran.key(0)

    m_obs = jnp.array([11.0, 12.0])
    t_obs = jnp.array([13.0, 13.5])

    n_sample = 1000
    n_t = 100
    t_grid = np.linspace(0.5, 13.5, n_t)
    logt0 = jnp.log10(13.8)

    log_mah, t_grid = mc_mah_cenpop(
        ran_key,
        m_obs,
        t_obs,
        t_grid,
        logt0,
        n_sample=n_sample,
        params=DEFAULT_DIFFMAHPOP_PARAMS,
        n_t=n_t,
        return_mah_params=False,
    )

    assert np.all(np.isfinite(log_mah))
    assert np.all(np.isfinite(t_grid))
    assert log_mah.shape == (m_obs.size * n_sample, n_t)
    assert t_grid.size == n_t
