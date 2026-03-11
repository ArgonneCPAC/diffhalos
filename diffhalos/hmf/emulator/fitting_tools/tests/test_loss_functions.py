""" """

import numpy as np

from .. import loss_functions as lf
from ....hmf_model import predict_diff_hmf
from ....hmf_param_utils import (
    DEFAULT_HMF_PARAMS,
    N_DIFFSKY_HMF_PARAMS,
    define_diffsky_hmf_params_array_from_namedtuple,
    define_diffsky_hmf_params_namedtuple_from_array,
)


def test_mse_works_as_expected():

    pred = np.linspace(0, 100, 100)
    target = np.linspace(0, 100, 100)

    mse = lf.mse(pred, target)
    assert mse == 0

    pred = np.linspace(0, 100, 100)
    target = np.linspace(0, 100, 100) + 1

    mse = lf.mse(pred, target)
    assert mse == 1


def test_mse_loss_diff_hmf_curve_computes():

    n_cosmo = 10

    preds = np.zeros((n_cosmo, N_DIFFSKY_HMF_PARAMS))
    perturb = np.random.uniform(low=-0.2, high=0.2, size=n_cosmo)

    for i in range(n_cosmo):
        params_array = define_diffsky_hmf_params_array_from_namedtuple(
            DEFAULT_HMF_PARAMS
        )
        params_array = params_array.at[0].set(params_array.at[0].get() + perturb[i])
        preds[i] = params_array

    n_halo = 100
    n_z = 3

    z = np.array([0.5, 1.0, 2.0])
    lgmp_arr = np.linspace(10.0, 14.0, n_halo)

    target = np.zeros((n_cosmo, n_z, n_halo))
    for ic in range(n_cosmo):
        for iz in range(n_z):
            params_ntup = define_diffsky_hmf_params_namedtuple_from_array(preds[ic])
            target[ic, iz, :] = predict_diff_hmf(params_ntup, lgmp_arr, z[iz])

    lgmp = np.repeat(
        np.repeat(lgmp_arr[np.newaxis, :], n_z, axis=0)[np.newaxis, ...],
        n_cosmo,
        axis=0,
    )
    loss_val = lf.mse_loss_diff_hmf_curve(preds, target, lgmp, z)

    assert np.isfinite(loss_val)
    assert loss_val > 0.0
    assert np.allclose(loss_val, 0.0)
