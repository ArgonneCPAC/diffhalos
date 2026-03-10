""""""

import numpy as np
import os

from .. import training_data_generator as tdg
from .....cosmology.cosmo_param_utils import DEFAULT_COSMO_PRIORS, DEFAULT_COSMOLOGY

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(HERE, "testing_data")
SAVE_BASE_NAME_DIFF = "testing_diff"
SAVE_BASE_NAME_CUML = "testing_cuml"


def test_generate_diff_hmf_loss_train_data():
    logmp = np.linspace(8.5, 15.5, 100)
    z = np.array([0.0, 1.5, 2.5, 3.5, 5.0])

    hmf_cut = 1e-8
    num_samples = 3

    loss_data = tdg.generate_hmf_loss_train_data(
        logmp,
        z,
        cuml=False,
        base_cosmo_params=DEFAULT_COSMOLOGY,
        cosmo_priors=DEFAULT_COSMO_PRIORS,
        num_samples=num_samples,
        hmf_cut=hmf_cut,
        savedir=SAVE_DIR,
        save_base_name=SAVE_BASE_NAME_DIFF,
        return_outputs=True,
    )

    assert len(loss_data) == num_samples
    for ci in range(len(loss_data)):
        for zi in range(len(z)):
            assert np.all(np.isfinite(loss_data[ci][zi][0]))
            assert np.all(np.isfinite(loss_data[ci][zi][1]))
            assert np.all(np.isfinite(loss_data[ci][zi][2]))
            assert loss_data[ci][zi][0] == z[zi]

    # check that loading the loss data works without issues
    loss_data = tdg.load_hmf_loss_data(
        savedir=SAVE_DIR,
        save_base_name=SAVE_BASE_NAME_DIFF,
        cuml=False,
        return_outputs=True,
    )

    res = tdg.generate_best_fit_hmf_params_train_data(
        loss_data,
        num_steps=100,
        step_size=1e-3,
        n_warmup=5,
        cuml=False,
        savedir=SAVE_DIR,
        save_base_name=SAVE_BASE_NAME_DIFF,
        return_outputs=True,
    )

    for _res in res:
        p_best, loss, loss_hist, params_hist, fit_terminates = _res
        assert np.all(
            np.isfinite(
                np.concatenate(
                    [np.asarray(getattr(p_best, _field)) for _field in p_best._fields]
                )
            )
        )
        assert np.all(np.isfinite(loss))
        assert np.all(np.isfinite(loss_hist))
        assert fit_terminates == 1
        assert loss < loss_hist[0]


def test_generate_cuml_hmf_loss_train_data():
    logmp = np.linspace(8.5, 15.5, 100)
    z = np.array([0.0, 1.5, 2.5, 3.5, 5.0])

    hmf_cut = 1e-8
    num_samples = 3

    loss_data = tdg.generate_hmf_loss_train_data(
        logmp,
        z,
        cuml=True,
        base_cosmo_params=DEFAULT_COSMOLOGY,
        cosmo_priors=DEFAULT_COSMO_PRIORS,
        num_samples=num_samples,
        hmf_cut=hmf_cut,
        savedir=SAVE_DIR,
        save_base_name=SAVE_BASE_NAME_CUML,
        return_outputs=True,
    )

    assert len(loss_data) == num_samples
    for ci in range(len(loss_data)):
        for zi in range(len(z)):
            assert np.all(np.isfinite(loss_data[ci][zi][0]))
            assert np.all(np.isfinite(loss_data[ci][zi][1]))
            assert np.all(np.isfinite(loss_data[ci][zi][2]))
            assert loss_data[ci][zi][0] == z[zi]

    # check that loading the loss data works without issues
    loss_data = tdg.load_hmf_loss_data(
        savedir=SAVE_DIR,
        save_base_name=SAVE_BASE_NAME_CUML,
        cuml=True,
        return_outputs=True,
    )
    assert len(loss_data) == num_samples
    for ci in range(len(loss_data)):
        for zi in range(len(z)):
            assert np.all(np.isfinite(loss_data[ci][zi][0]))
            assert np.all(np.isfinite(loss_data[ci][zi][1]))
            assert np.all(np.isfinite(loss_data[ci][zi][2]))
            assert loss_data[ci][zi][0] == z[zi]

    res = tdg.generate_best_fit_hmf_params_train_data(
        loss_data,
        num_steps=100,
        step_size=1e-3,
        n_warmup=5,
        cuml=True,
        savedir=SAVE_DIR,
        save_base_name=SAVE_BASE_NAME_CUML,
        return_outputs=True,
    )

    for _res in res:
        p_best, loss, loss_hist, params_hist, fit_terminates = _res
        assert np.all(
            np.isfinite(
                np.concatenate(
                    [np.asarray(getattr(p_best, _field)) for _field in p_best._fields]
                )
            )
        )
        assert np.all(np.isfinite(loss))
        assert np.all(np.isfinite(loss_hist))
        assert fit_terminates == 1
        assert loss < loss_hist[0]


def test_generate_hmf_loss_train_data_unpacks_properly():
    logmp = np.linspace(8.5, 15.5, 100)
    z = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    num_z = len(z)

    # for single cosmology
    num_samples = 1
    loss_data = tdg.generate_hmf_loss_train_data(
        logmp,
        z,
        cuml=True,
        cosmo_params=None,
        cosmo_param_names=None,
        base_cosmo_params=DEFAULT_COSMOLOGY,
        cosmo_priors=DEFAULT_COSMO_PRIORS,
        num_samples=num_samples,
        savedir=None,
        save_base_name=None,
        return_outputs=True,
    )

    # one cosmology, many redshift
    assert len(loss_data) == num_z
    for _loss_z in loss_data:
        try:
            redshift, target_lgmp, target_hmf = _loss_z
        except ValueError:
            raise Exception(
                "Loss data for each cosmology must be a list of len=3,\n but it has len = %d"
                % len(loss_data),
            )

    # multiple cosmologies, many redshifts
    num_samples = 5
    loss_data = tdg.generate_hmf_loss_train_data(
        logmp,
        z,
        cuml=True,
        cosmo_params=None,
        cosmo_param_names=None,
        base_cosmo_params=DEFAULT_COSMOLOGY,
        cosmo_priors=DEFAULT_COSMO_PRIORS,
        num_samples=num_samples,
        savedir=None,
        save_base_name=None,
        return_outputs=True,
    )

    assert len(loss_data) == num_samples
    for _loss in loss_data:
        assert len(_loss) == num_z
        for _loss_z in _loss:
            try:
                redshift, target_lgmp, target_hmf = _loss_z
            except ValueError:
                raise Exception(
                    "Loss data for each cosmology must be a list of len=3,\n but it has len = %d"
                    % len(loss_data),
                )
