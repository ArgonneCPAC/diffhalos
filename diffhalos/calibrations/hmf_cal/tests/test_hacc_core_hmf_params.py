""" """

import numpy as np

from .. import hacc_core_hmf_params as hchmf


def test_hmf_params():
    hmf_params = hchmf.HMF_PARAMS
    for params in hmf_params:
        assert np.all(np.isfinite(params))
