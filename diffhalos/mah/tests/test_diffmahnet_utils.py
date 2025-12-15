""" """

import numpy as np
from jax import random as jran
from jax import numpy as jnp

from ..diffmahnet_utils import mc_mah_cenpop, mc_mah_satpop, get_mean_and_std_of_mah
from ..diffmahpop_utils import mc_mah_cenpop as mc_mah_cenpop_diffmahpop


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

    cen_mah, _ = mc_mah_cenpop(
        m_vals,
        t_vals,
        ran_key,
        t_grid,
        logt0=logt0,
    )

    assert np.all(np.isfinite(cen_mah))
    assert cen_mah.shape == t_grid.shape == (n_cens * n_sample, n_t)


def test_cenpop_diffmahnet_vs_diffmahpop():

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

    cen_mah, _ = mc_mah_cenpop(
        m_vals,
        t_vals,
        ran_key,
        t_grid,
        logt0=logt0,
    )

    t_grid = jnp.linspace(t_min, t_obs.max(), n_t)

    cen_mah_diffmahpop, _ = mc_mah_cenpop_diffmahpop(
        m_vals,
        t_vals,
        ran_key,
        t_grid,
        logt0=logt0,
    )

    assert np.all(np.isfinite(cen_mah))
    assert np.all(np.isfinite(cen_mah_diffmahpop))

    mah_mean = get_mean_and_std_of_mah(cen_mah)[0]
    mah_mean_diffmahpop = get_mean_and_std_of_mah(cen_mah_diffmahpop)[0]

    assert np.allclose(mah_mean, mah_mean_diffmahpop, rtol=0.1)


def test_mc_mah_satpop_behaves_as_expected():

    ran_key = jran.key(0)

    n_subs = 10
    m_obs = np.linspace(9.0, 14.0, n_subs)
    t_obs = np.linspace(12.0, 13.5, n_subs)

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

    sat_mah, _ = mc_mah_satpop(
        m_vals,
        t_vals,
        ran_key,
        t_grid,
        logt0=logt0,
    )

    assert np.all(np.isfinite(sat_mah))
    assert sat_mah.shape == t_grid.shape == (n_subs * n_sample, n_t)
