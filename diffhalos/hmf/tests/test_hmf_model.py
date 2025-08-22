""" """

import numpy as np

from ..hmf_model import (
    predict_cuml_hmf,
    predict_differential_hmf,
    DEFAULT_HMF_PARAMS,
)


def test_cuml_hmf_evaluations():
    lgmp_arr = np.linspace(-6, 0, 500)
    redshift = 0.2
    res = predict_cuml_hmf(DEFAULT_HMF_PARAMS, lgmp_arr, redshift)
    assert res.shape == lgmp_arr.shape
    assert np.all(np.isfinite(res))


def test_diff_hmf_evaluations():
    lgmp_arr = np.linspace(-6, 0, 500)
    redshift = 0.2
    res = predict_differential_hmf(DEFAULT_HMF_PARAMS, lgmp_arr, redshift)
    assert res.shape == lgmp_arr.shape
    assert np.all(np.isfinite(res))
