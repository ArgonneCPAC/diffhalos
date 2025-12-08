""" """

import numpy as np

from ..default_params import DEFAULT_CCSHMF_PARAMS as ccshmf_params


def test_ccshmf_params():
    for params in ccshmf_params:
        assert np.all(np.isfinite(params))
