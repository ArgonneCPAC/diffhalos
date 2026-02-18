""""""

import numpy as np

from ..training_data_generator import (
    get_hmf_training_data,
)

from ...param_utils.cosmo_params import DEFAULT_COSMO_PRIORS, DEFAULT_COSMOLOGY


def test_get_diff_hmf_training_data():
    logMhalo = np.linspace(8.5, 15.5, 100)
    z = np.array([0.0, 1.5, 2.5, 3.5, 5.0])

    hmf_cut = 1e-8
    num_samples = 10

    loss_data = get_hmf_training_data(
        logMhalo,
        z,
        cuml=False,
        base_cosmo_params=DEFAULT_COSMOLOGY,
        cosmo_priors=DEFAULT_COSMO_PRIORS,
        num_samples=num_samples,
        hmf_cut=hmf_cut,
        savedir=None,
        save_base_name=None,
        return_outputs=True,
    )

    assert len(loss_data) == num_samples
    for ci in range(len(loss_data)):
        for zi in range(len(z)):
            assert np.all(np.isfinite(loss_data[ci][zi][0]))
            assert np.all(np.isfinite(loss_data[ci][zi][1]))
            assert np.all(np.isfinite(loss_data[ci][zi][2]))
            assert loss_data[ci][zi][0] == z[zi]


def test_get_cuml_hmf_training_data():
    logMhalo = np.linspace(8.5, 15.5, 100)
    z = np.array([0.0, 1.5, 2.5, 3.5, 5.0])

    hmf_cut = 1e-8
    num_samples = 10

    loss_data = get_hmf_training_data(
        logMhalo,
        z,
        cuml=True,
        base_cosmo_params=DEFAULT_COSMOLOGY,
        cosmo_priors=DEFAULT_COSMO_PRIORS,
        num_samples=num_samples,
        hmf_cut=hmf_cut,
        savedir=None,
        save_base_name=None,
        return_outputs=True,
    )

    assert len(loss_data) == num_samples
    for ci in range(len(loss_data)):
        for zi in range(len(z)):
            assert np.all(np.isfinite(loss_data[ci][zi][0]))
            assert np.all(np.isfinite(loss_data[ci][zi][1]))
            assert np.all(np.isfinite(loss_data[ci][zi][2]))
            assert loss_data[ci][zi][0] == z[zi]
