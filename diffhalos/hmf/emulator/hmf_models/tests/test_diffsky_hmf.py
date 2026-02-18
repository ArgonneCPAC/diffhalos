""""""

import numpy as np

from ..diffsky_hmf import (
    diffsky_cuml_hmf,
    diffsky_diff_hmf,
    get_binned_diff_from_cuml_hmf,
)

from diffsky.mass_functions.hmf_calibrations import DEFAULT_HMF_PARAMS


def test_diffsky_cuml_hmf():
    logmp = np.linspace(9.0, 14.0, 100)
    redshift = 0.3

    log_cuml_hmf = diffsky_cuml_hmf(
        DEFAULT_HMF_PARAMS,
        logmp,
        redshift,
    )

    assert np.all(np.isfinite(log_cuml_hmf))


def test_diffsky_diff_hmf():
    logmp = np.linspace(9.0, 14.0, 100)
    redshift = 0.3

    log_diff_hmf = diffsky_diff_hmf(
        DEFAULT_HMF_PARAMS,
        logmp,
        redshift,
    )

    assert np.all(np.isfinite(log_diff_hmf))


def test_get_binned_diff_from_cuml_hmf():
    logmp = np.linspace(9.0, 14.0, 100)
    redshift = 0.3

    binned_cuml_hmf = get_binned_diff_from_cuml_hmf(
        DEFAULT_HMF_PARAMS,
        logmp,
        redshift,
        n_halo=800_000,
        n_bins=50,
        volume=1.0,
        n_tot=100,
    )

    assert np.all(np.isfinite(binned_cuml_hmf))
