""""""

import numpy as np


from ..hmf_diff_fitter import diff_hmf_fitter
from ..training_data_generator import generate_hmf_loss_train_data
from .....cosmology.defaults import DEFAULT_COSMO_PRIORS, DEFAULT_COSMOLOGY


def test_diff_hmf_fitter_runs():
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
    res = diff_hmf_fitter(
        loss_data,
        n_steps=100,
        step_size=0.02,
        n_warmup=1,
    )
    p_best, loss, loss_hist, params_hist, fit_terminates = res

    assert np.all(np.isfinite(loss_hist))
    assert loss < loss_hist[0]
    assert fit_terminates == 1
