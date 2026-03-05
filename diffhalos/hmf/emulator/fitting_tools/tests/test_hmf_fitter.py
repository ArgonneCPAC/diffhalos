""""""

import numpy as np


from ..hmf_fitter import fit_hmf_single_cosmo, fit_hmf_multi_cosmo
from ..training_data_generator import generate_hmf_loss_train_data
from .....cosmology.cosmo_param_utils import sample_cosmo_params
from .....cosmology.cosmo import DEFAULT_COSMO_PRIORS, DEFAULT_COSMOLOGY


def test_adam_cuml_hmf_single_cosmo_fitter():
    logmp = np.linspace(8.5, 15.5, 100)
    z = np.array([0.0, 1.5, 2.5, 3.5, 5.0])

    loss_data = generate_hmf_loss_train_data(
        logmp,
        z,
        cuml=True,
        cosmo_params=None,
        cosmo_param_names=None,
        base_cosmo_params=DEFAULT_COSMOLOGY,
        cosmo_priors=DEFAULT_COSMO_PRIORS,
        num_samples=1,
        savedir=None,
        save_base_name=None,
        return_outputs=True,
    )
    res = fit_hmf_single_cosmo(
        loss_data,
        n_steps=100,
        step_size=0.02,
        n_warmup=1,
        cuml=True,
    )
    p_best, loss, loss_hist, params_hist, fit_terminates = res

    assert np.all(np.isfinite(loss_hist))
    assert loss < loss_hist[0]
    assert fit_terminates == 1


def test_adam_diff_hmf_single_cosmo_fitter():
    logmp = np.linspace(8.5, 15.5, 100)
    z = np.array([0.0, 1.5, 2.5, 3.5, 5.0])

    loss_data = generate_hmf_loss_train_data(
        logmp,
        z,
        cuml=False,
        cosmo_params=None,
        cosmo_param_names=None,
        base_cosmo_params=DEFAULT_COSMOLOGY,
        cosmo_priors=DEFAULT_COSMO_PRIORS,
        num_samples=1,
        savedir=None,
        save_base_name=None,
        return_outputs=True,
    )
    res = fit_hmf_single_cosmo(
        loss_data,
        n_steps=100,
        step_size=0.02,
        n_warmup=1,
        cuml=False,
    )
    p_best, loss, loss_hist, params_hist, fit_terminates = res

    assert np.all(np.isfinite(loss_hist))
    assert loss < loss_hist[0]
    assert fit_terminates == 1


def test_adam_diff_hmf_multi_cosmo_fitter():
    logmp = np.linspace(8.5, 15.5, 100)
    z = np.array([0.0, 1.5, 2.5, 3.5, 5.0])

    cosmo_params = sample_cosmo_params(
        cosmo_priors=DEFAULT_COSMO_PRIORS,
        seed=None,
        num_samples=2,
    )
    cosmo_param_names = list(DEFAULT_COSMO_PRIORS.keys())

    loss_data = generate_hmf_loss_train_data(
        logmp,
        z,
        cosmo_params=cosmo_params,
        cosmo_param_names=cosmo_param_names,
        cuml=False,
    )
    res = fit_hmf_multi_cosmo(
        loss_data,
        n_steps=100,
        step_size=0.02,
        n_warmup=1,
        cuml=False,
    )

    assert len(res) == cosmo_params.shape[0]
    for _res in res:
        p_best, loss, loss_hist, params_hist, fit_terminates = _res

        assert np.all(np.isfinite(loss_hist))
        assert loss < loss_hist[0]
        assert fit_terminates == 1
