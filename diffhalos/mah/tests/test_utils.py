""" """

import numpy as np

from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS

from ..utils import rescale_mah_parameters


def test_rescale_mah_parameters():

    mah_params_uncorrected = DEFAULT_MAH_PARAMS

    delta_mobs = 0.2
    logm_obs_uncorrected = mah_params_uncorrected.logm0
    logm_obs_corrected = logm_obs_uncorrected - delta_mobs

    mah_params_corrected = rescale_mah_parameters(
        mah_params_uncorrected,
        logm_obs_corrected,
        logm_obs_uncorrected,
    )

    for _field in mah_params_corrected._fields:
        assert np.all(np.isfinite(mah_params_corrected._asdict()[_field]))

    assert np.allclose(
        float(mah_params_uncorrected.logm0 - mah_params_corrected.logm0),
        delta_mobs,
    )
