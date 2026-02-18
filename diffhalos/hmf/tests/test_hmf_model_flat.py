""" """

import numpy as np

from .. import hmf_model_flat


def test_lg_hmf_kern_evaluates():
    lgmp_arr = np.linspace(10, 15, 500)
    redshift = 0.0
    hmf = hmf_model_flat.predict_cuml_hmf(
        hmf_model_flat.DEFAULT_FLAT_HMF_PARAMS, lgmp_arr, redshift
    )
    assert hmf.shape == hmf.shape
    assert np.all(np.isfinite(hmf))


def test_predict_hmf_returns_finite_valued_expected_shape():
    redshift = 1.0
    nhalos = 100
    lgmp_arr = np.linspace(10, 15, nhalos)
    pred = hmf_model_flat.predict_cuml_hmf(
        hmf_model_flat.DEFAULT_FLAT_HMF_PARAMS, lgmp_arr, redshift
    )
    assert pred.shape == lgmp_arr.shape
    assert np.all(np.isfinite(pred))

    lgmp = 12.0
    pred = hmf_model_flat.predict_cuml_hmf(
        hmf_model_flat.DEFAULT_FLAT_HMF_PARAMS, lgmp, redshift
    )
    assert pred.shape == ()
    assert np.all(np.isfinite(pred))

    nhalos = 5
    zarr = np.linspace(0, 5, nhalos)
    lgmp = 12.0
    pred = hmf_model_flat.predict_cuml_hmf(
        hmf_model_flat.DEFAULT_FLAT_HMF_PARAMS, lgmp, zarr
    )
    assert pred.shape == (nhalos,)
    assert np.all(np.isfinite(pred))

    nhalos = 5
    zarr = np.linspace(0, 5, nhalos)
    lgmp_arr = np.linspace(10, 15, nhalos)
    pred = hmf_model_flat.predict_cuml_hmf(
        hmf_model_flat.DEFAULT_FLAT_HMF_PARAMS, lgmp_arr, zarr
    )
    assert pred.shape == (nhalos,)
    assert np.all(np.isfinite(pred))


def test_cuml_hmf_evaluations():
    lgmp_arr = np.linspace(-6, 0, 500)
    redshift = 0.2
    res = hmf_model_flat.predict_cuml_hmf(
        hmf_model_flat.DEFAULT_FLAT_HMF_PARAMS,
        lgmp_arr,
        redshift,
    )
    assert res.shape == lgmp_arr.shape
    assert np.all(np.isfinite(res))


def test_diff_hmf_evaluations():
    lgmp_arr = np.linspace(-6, 0, 500)
    redshift = 0.2
    res = hmf_model_flat.predict_diff_hmf(
        hmf_model_flat.DEFAULT_FLAT_HMF_PARAMS,
        lgmp_arr,
        redshift,
    )
    assert res.shape == lgmp_arr.shape
    assert np.all(np.isfinite(res))
