""" """

import os
from glob import glob

import numpy as np

from ..ccshmf_model import (
    DEFAULT_CCSHMF_PARAMS,
    predict_ccshmf,
    predict_ccshmf_halopop,
    predict_differential_cshmf,
    predict_differential_cshmf_halopop,
    compute_mean_subhalo_counts,
    subhalo_lightcone_weights,
    N_LGMU_TABLE,
)

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
TESTING_DATA_DRN = os.path.join(_THIS_DRNAME, "testing_data")
BNPAT = "smdpl_cshmf_cuml_redshift_{0:.2f}_lgmhost_{1:.2f}.txt"


def _mse(pred, target):
    diff = pred - target
    return np.mean(diff**2)


def _mae(pred, target):
    diff = pred - target
    return np.mean(np.abs(diff))


def infer_redshift_from_bname(bn):
    return float(bn.split("_")[4])


def infer_logmhost_from_bname(bn):
    return float(bn.split("_")[-1][:-4])


def test_predict_ccshmf_returns_finite_valued_expected_shape():
    lgmhost = 13.0
    nsubs = 100
    lgmuarr = np.linspace(-5, 0, nsubs)
    pred = predict_ccshmf(DEFAULT_CCSHMF_PARAMS, lgmhost, lgmuarr)
    assert pred.shape == lgmuarr.shape
    assert np.all(np.isfinite(pred))

    lgmu = -2.0
    pred = predict_ccshmf(DEFAULT_CCSHMF_PARAMS, lgmhost, lgmu)
    assert pred.shape == ()
    assert np.all(np.isfinite(pred))

    nhosts = 5
    lgmhostarr = np.linspace(12, 15, nhosts)
    pred = predict_ccshmf(DEFAULT_CCSHMF_PARAMS, lgmhostarr, lgmu)
    assert pred.shape == (nhosts,)
    assert np.all(np.isfinite(pred))

    nhosts = 5
    nsubs = nhosts
    lgmhostarr = np.linspace(12, 15, nhosts)
    lgmuarr = np.linspace(-5, 0, nsubs)
    pred = predict_ccshmf(DEFAULT_CCSHMF_PARAMS, lgmhostarr, lgmuarr)
    assert pred.shape == (nhosts,)
    assert np.all(np.isfinite(pred))


def test_predict_ccshmf_halopop_returns_expected_shape():
    lgmhost = np.array([11.0, 13.0, 9.0])
    nsubs = 100
    lgmuarr = np.linspace(-5, 0, nsubs)
    pred = predict_ccshmf_halopop(DEFAULT_CCSHMF_PARAMS, lgmhost, lgmuarr)
    assert pred.shape == (lgmhost.size, nsubs)
    assert np.all(np.isfinite(pred))


def test_predict_differential_cshmf_returns_expected_shape():
    lgmhost = 11.0
    nsubs = 100
    lgmuarr = np.linspace(-5, 0, nsubs)
    pred = predict_differential_cshmf(DEFAULT_CCSHMF_PARAMS, lgmhost, lgmuarr)
    assert pred.shape == lgmuarr.shape
    assert np.all(np.isfinite(pred))


def test_predict_differential_cshmf_halopop_returns_expected_shape():
    lgmhost = np.array([11.0, 13.0, 9.0])
    nsubs = 100
    lgmuarr = np.linspace(-5, 0, nsubs)
    pred = predict_differential_cshmf_halopop(DEFAULT_CCSHMF_PARAMS, lgmhost, lgmuarr)
    assert pred.shape == (lgmhost.size, nsubs)
    assert np.all(np.isfinite(pred))


def test_predict_ccshmf_accurately_approximates_simulation_data():
    """This test loads some pretabulated CCSHMF data computed from SMPDL
    and compares the simulation results to the predict_ccshmf function"""
    fname_list = glob(os.path.join(TESTING_DATA_DRN, "smdpl_cshmf_*.txt"))
    bname_list = [os.path.basename(fn) for fn in fname_list]

    zlist = np.unique([infer_redshift_from_bname(bn) for bn in bname_list])

    for redshift in zlist:
        zpat = "redshift_{:.2f}".format(redshift)
        bname_list_z = [bn for bn in bname_list if zpat in bn]
        lgmh_list_z = np.array([infer_logmhost_from_bname(bn) for bn in bname_list_z])
        lgmh_list_z = lgmh_list_z[lgmh_list_z > 12]
        lgmhost_targets = np.sort(lgmh_list_z)

        for itarget in range(lgmhost_targets.size):
            target_lgmhost = lgmhost_targets[itarget]
            bn_sample = BNPAT.format(redshift, target_lgmhost)
            cshmf_data_sample = np.loadtxt(os.path.join(TESTING_DATA_DRN, bn_sample))
            target_lgmu_bins, target_lg_ccshmf = (
                cshmf_data_sample[:, 0],
                cshmf_data_sample[:, 1],
            )
            pred_lg_ccshmf = predict_ccshmf(
                DEFAULT_CCSHMF_PARAMS, target_lgmhost, target_lgmu_bins
            )

            loss_sq = _mse(pred_lg_ccshmf, target_lg_ccshmf)
            assert np.sqrt(loss_sq) < 0.15

            loss_mae = _mae(pred_lg_ccshmf, target_lg_ccshmf)
            assert loss_mae < 0.06


def test_compute_mean_subhalo_counts():
    lgmhost = np.array([11.0, 13.0, 9.0])
    lgmp_min = 12.0
    nsubs = compute_mean_subhalo_counts(
        lgmhost,
        lgmp_min,
        ccshmf_params=DEFAULT_CCSHMF_PARAMS,
    )
    assert np.all(np.isfinite(nsubs))


def test_subhalo_lightcone_weights():
    lgmhost = np.array([11.0, 13.0, 9.0])
    lgmp_min = 12.0
    weights, lgmu = subhalo_lightcone_weights(
        lgmhost,
        lgmp_min,
        DEFAULT_CCSHMF_PARAMS,
    )
    assert weights.shape == (lgmhost.size, N_LGMU_TABLE)
    assert np.all(np.isfinite(weights))

    assert lgmu.shape == (lgmhost.size, N_LGMU_TABLE)
    assert np.all(np.isfinite(lgmu))
