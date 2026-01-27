""" """

import numpy as np
from jax import random as jran


from ..hmf_model import (
    DEFAULT_HMF_PARAMS,
    predict_diff_hmf,
)
from ..mc_hosts import (
    LGMH_MAX,
    mc_host_halos_singlez,
    mc_host_halos_hist_singlez,
)


def test_mc_host_halo_logmp_behaves_as_expected():
    ran_key = jran.PRNGKey(0)

    lgmp_min = 11.0
    redshift = 0.1
    Lbox = 1000.0
    volume_com_mpc = Lbox**3

    lgmp_halopop = mc_host_halos_singlez(
        ran_key,
        lgmp_min,
        redshift,
        volume_com_mpc,
    )
    assert lgmp_halopop.size > 0
    assert np.all(lgmp_halopop > lgmp_min)
    assert np.all(lgmp_halopop < LGMH_MAX)

    lgmp_max = 14.0
    lgmp_halopop2 = mc_host_halos_singlez(
        ran_key,
        lgmp_min,
        redshift,
        volume_com_mpc,
        lgmp_max=lgmp_max,
    )
    assert np.all(lgmp_halopop > lgmp_min)
    assert np.all(lgmp_halopop2 < lgmp_max)


def test_diff_hmf_pophist_vs_theory_consistency():
    ran_key = jran.key(0)
    lgmp_min = 11.0
    Lbox = 1000.0
    Vbox = Lbox**3

    lgm_bins = np.linspace(lgmp_min + 0.5, 14.0, 50)

    z_test = np.linspace(0.0, 1.0, 5)

    for z in z_test:
        diff_hmf = predict_diff_hmf(DEFAULT_HMF_PARAMS, lgm_bins, z)

        diff_hmf_target, lgm_binmids = mc_host_halos_hist_singlez(
            ran_key, lgmp_min, z, Vbox, bins=lgm_bins
        )

        # interpolate to compare same-sized arrays
        diff_hmf_interp = 10 ** np.interp(lgm_binmids, lgm_bins, diff_hmf)

        assert np.allclose(diff_hmf_interp, diff_hmf_target, rtol=0.1)
