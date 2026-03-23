""""""

import numpy as np

from ..hmf_cuml_fitter import cuml_hmf_fitter
from ..training_data_generator import generate_hmf_loss_train_data
from .....cosmology.cosmo import DEFAULT_COSMO_PRIORS, DEFAULT_COSMOLOGY_DICT


def test_cuml_hmf_fitter_single_cosmo():
    logmp = np.linspace(8.5, 15.5, 100)
    z = np.array([0.0, 1.5, 2.5, 3.5, 5.0])

    loss_data = generate_hmf_loss_train_data(
        logmp,
        z,
        cuml=True,
        cosmo_params=None,
        cosmo_param_names=None,
        base_cosmo_params=DEFAULT_COSMOLOGY_DICT,
        cosmo_priors=DEFAULT_COSMO_PRIORS,
        num_samples=1,
        savedir=None,
        save_base_name=None,
        return_outputs=True,
    )
    res = cuml_hmf_fitter(
        loss_data,
        n_steps=100,
        step_size=0.02,
        n_warmup=1,
    )
    p_best, loss, loss_hist, params_hist, fit_terminates = res

    assert np.all(np.isfinite(loss_hist))
    assert loss < loss_hist[0]
    assert fit_terminates == 1
