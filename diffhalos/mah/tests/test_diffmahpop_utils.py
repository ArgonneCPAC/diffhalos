""" """

import numpy as np
from jax import numpy as jnp
from jax import random as jran

from ..diffmahpop_utils import (
    mc_mah_cenpop,
    mc_mah_satpop,
    DEFAULT_DIFFMAHPOP_PARAMS,
)


def test_mc_mah_cenpop_behaves_as_expected():

    ran_key = jran.key(0)

    n_cens = 10
    m_obs = np.linspace(9.0, 14.0, n_cens)
    t_obs = np.linspace(12.0, 13.5, n_cens)

    n_sample = 1000
    n_t = 100
    t_min = 0.5
    logt0 = np.log10(13.8)

    # get a list of (m_obs, t_obs) for each MC realization
    m_vals, t_vals = [
        jnp.repeat(x.flatten(), n_sample)
        for x in np.stack(
            [m_obs, t_obs],
            axis=-1,
        ).T
    ]

    # construct time grids for each halo, given observation time
    t_grid = jnp.linspace(t_min, t_obs.max(), n_t)

    log_mah, _ = mc_mah_cenpop(
        m_vals,
        t_vals,
        ran_key,
        t_grid,
        logt0=logt0,
        params=DEFAULT_DIFFMAHPOP_PARAMS,
    )

    assert np.all(np.isfinite(log_mah))
    assert np.all(np.isfinite(t_grid))
    assert log_mah.shape == (m_obs.size * n_sample, n_t)
    assert t_grid.size == n_t


def test_mc_mah_satpop_behaves_as_expected():

    ran_key = jran.key(0)

    n_sats = 10
    m_obs = np.linspace(9.0, 14.0, n_sats)
    t_obs = np.linspace(12.0, 13.5, n_sats)

    n_sample = 1000
    n_t = 100
    t_min = 0.5
    logt0 = np.log10(13.8)

    # get a list of (m_obs, t_obs) for each MC realization
    m_vals, t_vals = [
        jnp.repeat(x.flatten(), n_sample)
        for x in np.stack(
            [m_obs, t_obs],
            axis=-1,
        ).T
    ]

    # construct time grids for each halo, given observation time
    t_grid = jnp.linspace(t_min, t_obs.max(), n_t)

    log_mah, _ = mc_mah_satpop(
        m_vals,
        t_vals,
        ran_key,
        t_grid,
        logt0=logt0,
        params=DEFAULT_DIFFMAHPOP_PARAMS,
    )

    assert np.all(np.isfinite(log_mah))
    assert np.all(np.isfinite(t_grid))
    assert log_mah.shape == (m_obs.size * n_sample, n_t)
    assert t_grid.size == n_t
