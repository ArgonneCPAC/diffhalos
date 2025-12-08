""" """

import numpy as np
from jax import random as jran
from jax import numpy as jnp

from ..diffmahnet_utils import mc_mah_cenpop


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
    t_grid = jnp.linspace(t_min, t_vals, n_t).T

    cen_mah, tgrid, _ = mc_mah_cenpop(
        m_vals,
        t_vals,
        ran_key,
        t_grid,
        centrals_model_key="cenflow_v1_0train_float64.eqx",
        logt0=logt0,
    )

    assert np.all(np.isfinite(cen_mah))
    assert np.all(np.isfinite(tgrid))
    assert cen_mah.shape == tgrid.shape == (n_cens * n_sample, n_t)
