""" """

import numpy as np

from ..ccshmf_kernels import (
    DEFAULT_CCSHMF_KERN_PARAMS,
    lg_cuml_cshmf_kern,
    lg_diff_cshmf_kern,
)


def test_lg_ccshmf_kern_evaluates():
    lgmu_arr = np.linspace(-6, 0, 500)
    res = lg_cuml_cshmf_kern(DEFAULT_CCSHMF_KERN_PARAMS, lgmu_arr)
    assert res.shape == lgmu_arr.shape
    assert np.all(np.isfinite(res))


def test_lg_differential_cshmf_kern_evaluates():
    lgmu_arr = np.linspace(-6, 0, 500)
    res = lg_diff_cshmf_kern(DEFAULT_CCSHMF_KERN_PARAMS, lgmu_arr)
    assert res.shape == lgmu_arr.shape
    assert np.all(np.isfinite(res))
