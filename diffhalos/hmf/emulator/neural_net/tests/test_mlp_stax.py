""" """

import numpy as np
import os

from ..mlp_stax import (
    MLP_stax,
    load_mlp_model,
    predict_mlp_hmf_params,
    DEFAULT_MLP_MODEL,
)

from ..pretrained_models.data_index import ENVIRON_VAR

from ...fitting_tools import hmf_fitter
from ...fitting_tools.loss_functions import mse
from ...fitting_tools import training_data_generator as tdg
from ....hmf_param_utils import define_diffsky_hmf_params_namedtuple_from_array
from ....hmf_model import predict_diff_hmf
from .....cosmology.cosmo_param_utils import sample_cosmo_params


def test_load_mlp_model_works():

    try:
        savedir = os.environ[ENVIRON_VAR]
        name = DEFAULT_MLP_MODEL
        mlp = load_mlp_model(savedir=savedir, name=name)
        assert mlp.load_successful
    except KeyError:
        pass


def test_predict_mlp_hmf_params_computes():
    try:
        savedir = os.environ[ENVIRON_VAR]
        name = DEFAULT_MLP_MODEL

        # Om0, sigma8 for model v0
        cosmo_params = np.array([0.3, 0.8])

        nn_hmf_params_ntup = predict_mlp_hmf_params(
            cosmo_params, savedir=savedir, name=name
        )

        assert isinstance(nn_hmf_params_ntup, tuple)

    except KeyError:
        pass


def create_test_mlp_training_data(logmp, z):

    num_samples = 5
    hmf_cut = 1e-8
    cuml = False
    num_steps = 1_000
    step_size = 3e-2
    n_warmup = 10

    cosmo_priors = {"Om0": (0.2, 0.5), "sigma8": (0.6, 1.0)}
    cosmo_param_names = cosmo_priors.keys()

    cosmo_params = sample_cosmo_params(
        cosmo_priors=cosmo_priors, num_samples=num_samples
    )

    # generate the hmf fitter input data
    loss_data = tdg.generate_hmf_loss_train_data(
        logmp,
        z,
        cosmo_params=cosmo_params,
        cosmo_param_names=cosmo_param_names,
        cuml=cuml,
        hmf_cut=hmf_cut,
        savedir=None,
        return_outputs=True,
    )

    # run the fitter to get the hmf best-fit parameters
    res = hmf_fitter.fit_hmf_multi_cosmo(
        loss_data,
        n_steps=num_steps,
        step_size=step_size,
        n_warmup=n_warmup,
        cuml=cuml,
    )

    num_samples = len(loss_data)

    ytp_params = np.zeros((num_samples, 5))
    x0_params = np.zeros((num_samples, 5))
    lo_params = np.zeros((num_samples, 4))
    hi_params = np.zeros((num_samples, 5))
    for ci in range(num_samples):
        p_best, loss, loss_hist, params_hist, fit_terminates = res[ci]

        ytp_params[ci, :] = np.asarray(p_best.ytp_params)
        x0_params[ci, :] = np.asarray(p_best.x0_params)
        lo_params[ci, :] = np.asarray(p_best.lo_params)
        hi_params[ci, :] = np.asarray(p_best.hi_params)

    hmf_params = np.zeros((num_samples, 19))
    for i in range(num_samples):
        hmf_params[i, :] = (
            list(ytp_params[i, :])
            + list(x0_params[i, :])
            + list(lo_params[i, :])
            + list(hi_params[i, :])
        )

    return hmf_params, cosmo_params, loss_data


def test_train_mlp():

    logmp = np.linspace(11.0, 15.0, 100)
    z = np.array([0.5, 1.0, 2.0])

    target_data, input_data, loss_data = create_test_mlp_training_data(logmp, z)

    # features and targets of the testing training data
    n_features, n_targets = input_data.shape[1], target_data.shape[1]
    hidden_layers = [8, 8, 16, 16, 8]
    SEED = 1291

    # initialize the MLP
    mlp = MLP_stax(n_targets, hidden_layers)
    _, net_apply = mlp.init_mlp()

    # sampler parameters
    step_size = 8e-4
    batch_size = input_data.shape[0]
    num_epochs = 100_000

    # initialize the optimizer
    _ = mlp.init_optimizer(step_size, n_features, seed=SEED)

    # train the model
    loss_history, opt_state = mlp.train_mlp(
        input_data,
        target_data,
        mse,
        num_epochs=num_epochs,
        batch_size=batch_size,
        timeit=False,
    )

    assert np.all(np.isfinite(loss_history))
    assert loss_history[0] > loss_history[-1]

    n_sam = input_data.shape[0]

    for i in range(n_sam):
        _loss = loss_data[i]

        _x = input_data[i, :]
        nn_hmf_params = net_apply(mlp.get_params(opt_state), _x).flatten()

        nn_hmf_params_ntup = define_diffsky_hmf_params_namedtuple_from_array(
            params=nn_hmf_params
        )

        for iz in range(len(_loss)):
            redshift, logmhalo, loghmf = _loss[iz]
            _y = predict_diff_hmf(nn_hmf_params_ntup, logmhalo, redshift)

            assert np.all(np.isfinite(_y))
            assert np.all(np.isclose(_y, loghmf, rtol=0.2))
