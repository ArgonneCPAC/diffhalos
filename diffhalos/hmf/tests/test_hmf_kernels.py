""" """

import numpy as np

from ..hmf_kernels import (
    DEFAULT_HMF_KERN_PARAMS,
    lg_hmf_kern,
    hmf_kern,
    lg_diff_hmf_kern,
)


def test_lg_hmf_kern_evaluates():
    lgmp_arr = np.linspace(-6, 0, 500)
    res = lg_hmf_kern(DEFAULT_HMF_KERN_PARAMS, lgmp_arr)
    assert res.shape == lgmp_arr.shape
    assert np.all(np.isfinite(res))


def test_hmf_kern_evaluates():
    lgmp_arr = np.linspace(-6, 0, 500)
    res = hmf_kern(DEFAULT_HMF_KERN_PARAMS, lgmp_arr)
    assert res.shape == lgmp_arr.shape
    assert np.all(np.isfinite(res))


def test_lg_diff_hmf_kern_evaluates():
    lgmp_arr = np.linspace(-6, 0, 500)
    res = lg_diff_hmf_kern(DEFAULT_HMF_KERN_PARAMS, lgmp_arr)
    assert res.shape == lgmp_arr.shape
    assert np.all(np.isfinite(res))
