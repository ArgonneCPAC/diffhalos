""" """

import numpy as np
from jax import random as jran

from ..diffmahnet_utils import mc_mah_cenpop


def test_mc_mah_cenpop_behaves_as_expected():

    ran_key = jran.key(0)

    n_cens = 10
    m_obs = np.linspace(9.0, 14.0, n_cens)
    t_obs = np.linspace(12.0, 13.5, n_cens)

    n_sample = 1000
    n_t = 100
    t_grid = np.linspace(0.5, 13.5, n_t)

    cen_mah, tgrid = mc_mah_cenpop(
        ran_key,
        m_obs,
        t_obs,
        t_grid,
        n_sample=n_sample,
        n_t=n_t,
        centrals_model_key="cenflow_v1_0train_float64.eqx",
    )

    assert np.all(np.isfinite(cen_mah))
    assert np.all(np.isfinite(tgrid))
    assert cen_mah.shape == tgrid.shape == (n_cens * n_sample, n_t)
