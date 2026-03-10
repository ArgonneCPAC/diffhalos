"""Building a MLP using jax's stax library and Adam optimizer"""

import numpy as np
from time import time
import pickle
import os
import pathlib
import warnings
from functools import partial

from jax import jit as jjit
from jax import random as jran
from jax import value_and_grad
from jax.example_libraries import stax
from jax.example_libraries import optimizers as jax_opt

from .pretrained_models import ENVIRON_VAR
from ..utils import format_time
from ...hmf_param_utils import define_diffsky_hmf_params_namedtuple

__all__ = ("MLP_stax", "load_mlp_model", "predict_mlp_hmf_params")


# load pretrained models
try:
    PRETRAINED_PATH = pathlib.Path(os.environ[ENVIRON_VAR])
    PRETRAINED_MODEL_NAMES = [
        f.name.split(".")[0]
        for f in PRETRAINED_PATH.iterdir()
        if f.is_file() and f.name.endswith(".pkl")
    ]
except KeyError:
    msg = (
        f"You have not set the '{ENVIRON_VAR}' environment variable to load HMF emulator models.\n"
        f"To set it, first run 'export {ENVIRON_VAR}=path_to_data_folder'.\n"
        "Otherwise, the full path to the models will have to be provided in `load_model(savedir=path_to_data_folder)`."
    )
    warnings.warn(msg, UserWarning)
    PRETRAINED_PATH = None
    PRETRAINED_MODEL_NAMES = []

# set the default model to use
DEFAULT_MLP_MODEL = "mlp_model_v0"


def load_mlp_model(savedir=PRETRAINED_PATH, name=DEFAULT_MLP_MODEL):
    """
    Load a `diffhmfemu` model

    Parameters
    ----------
    savedir: str
        path to directory where model is stored

    name: str
        base name for generated files

    Returns
    -------
    mlp: MLP_stax object
        instance of the MLP using the requested model
    """
    if name not in PRETRAINED_MODEL_NAMES:
        raise KeyError(
            f"Model {name} is not available\n"
            f"Choose one of {PRETRAINED_MODEL_NAMES}."
        )

    mlp = MLP_stax.load_model(
        savedir=savedir,
        save_base_name=name,
    )

    return mlp


@partial(jjit, static_argnames=["savedir", "name"])
def predict_mlp_hmf_params(
    cosmo_params,
    savedir=PRETRAINED_PATH,
    name=DEFAULT_MLP_MODEL,
):
    """
    Predict diffsky HMF paramters, given
    a set of cosmological parameters

    Parameters
    ----------
    cosmo_params: ndarray of shape (n_cosmo_params, )
        cosmological paramters

    savedir: str
        path to directory where model is stored

    name: str
        base name for generated files

    Returns
    -------
    nn_hmf_params_ntup: namedtuple
        mlp predicted HMF parameters at a given cosmology
        as a namedtuple compatible with the diffsky HMF model
    """
    mlp = load_mlp_model(savedir=savedir, name=name)

    nn_hmf_params = mlp.net_apply(
        mlp.get_params(mlp.opt_state_final), cosmo_params
    ).flatten()

    nn_hmf_params_ntup = define_diffsky_hmf_params_namedtuple(nn_hmf_params)

    return nn_hmf_params_ntup


class MLP_stax:

    def __init__(self, n_targets=19, hidden_layers=[8, 8, 16, 16, 8]):
        """
        Parameters
        ----------
        n_targets: int
            size of targets layer, by default 19
            which is the number of diffsky hmf parameters

        hidden_layers: list(int)
            number of neurons per hidden layer
        """

        self.n_targets = n_targets
        self.hidden_layers = hidden_layers

    def init_mlp(self, activation=stax.Selu):
        """
        Build the MLP using JAX's stax library.
        See https://docs.jax.dev/en/latest/jax.example_libraries.stax.html.

        Parameters
        ----------
        activation: function
            activation function to use;
            default is the Selu activation function

        Returns
        -------
        new layer, meaning an (init_fun, apply_fun) pair,
        representing the serial composition of the given sequence of layers;
        in particular:

        net_init: callable
            intilizes the MLP;
            takes as inputs (ran_key, input_shape=())

        net_apply: callable
            applies the neural network model
        """
        # hidden layers
        struct = ()
        for n_neurons in self.hidden_layers:
            struct += (stax.Dense(n_neurons), activation)

        # target layer
        self.net_init, self.net_apply = stax.serial(
            *struct,
            stax.Dense(self.n_targets),
        )

        self.n_targets = self.n_targets
        self.hidden_layers = self.hidden_layers

        return self.net_init, self.net_apply

    def init_optimizer(
        self,
        step_size,
        n_features,
        init_state=None,
        seed=0,
    ):
        """
        Initialize the optimizer

        Parameters
        ----------
        step_size: float
            step size to be used by the optimizer

        n_features: int
            number of features for building the nn model

        init_state: namedtuple
            initial state of the optimizer

        seed: int
            seed for rng generator,
            if ``ran_key`` is not provided

        Returns
        -------
        net_params_init: list
            initial state of neural network

        opt_init: funtion
            initializes the state of the optimizer

        opt_update: function
            update step function

        get_params: function
            returns the parameters as a list
        """
        self.step_size = step_size
        self.opt_seed = seed

        (
            self.opt_init,
            self.opt_update,
            self.get_params,
        ) = jax_opt.adam(step_size)

        ran_key = jran.PRNGKey(seed)
        ran_key, net_init_key = jran.split(ran_key)

        # features layer added
        if init_state is None:
            _, self.net_params_init = self.net_init(
                net_init_key, input_shape=(-1, n_features)
            )
            opt_state = self.opt_init(self.net_params_init)
        else:
            opt_state = self.opt_init(init_state)
            self.net_params_init = init_state
        self.opt_state_init = opt_state

        return (
            self.net_params_init,
            self.opt_init,
            self.opt_update,
            self.get_params,
        )

    def train_mlp(
        self,
        input_data,
        target_data,
        loss_function,
        loss_args=(),
        batch_size=None,
        num_batches=None,
        num_epochs=None,
        seed=0,
        timeit=False,
    ):
        """
        Run the loop over batches of training data to
        train the MLP model parameters

        Parameters
        ----------
        input_data: ndarray of shape (n_samples, n_features)
            input data to train on

        target_data: ndarray of shape (n_samples, n_target)
            target data to compare against;
            if None, the first item in ``self.loss_args`` is used

        loss_function: callable
            loss function to optimize

        loss_args: tuple
            arguments to pass to `loss_function`

        batch_size: int
            size of a batch

        num_batches: int
            number of batches

        num_epochs: int
            number of epochs for training

        seed: int
            random seed for batch-training

        timeit: bool
            if True, the training will be timed

        Returns
        -------
        loss_history: list
            loss value at each step of the training process
        """
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_epochs = num_epochs

        @jjit
        def _loss_fun(net_params, input_data, target_data):
            """
            Helper function for calling the
            loss function in the appropriate way

            Parameters
            ----------
            net_params: namedtuple or ndarray of shape (n_params, )
                optimizer's current state

            input_data: ndarray of shape (n_features, )
                input data for model calculation

            target_data: ndarray of shape (n_target, )
                target data for loss calculation
                against current optimizer's state

            Returns
            -------
            loss: float
                value of loss
            """
            preds = self.net_apply(net_params, input_data)
            loss = loss_function(preds, target_data, *loss_args)

            return loss

        @jjit
        def _train_step(
            step_i,
            opt_state,
            input_data,
            target_data,
        ):
            """
            Step function for updating parameters

            Parameters
            ----------
            step_i: int
                current step in gradient decent

            opt_state: namedtuple or ndarray of shape (n_params,)
                current optimizer state

            input_data: ndarray of shape (n_features,)
                input data for model calculation

            target_data: ndarray of shape (n_target,)
                target data for loss calculation
                against current optimizer's state

            Returns
            -------
            loss: float
                value of loss

            opt_update_state: namedtuple or ndarray of shape (n_params,)
                updated parameters of new optimizer's state
            """
            net_params = self.get_params(opt_state)
            loss, grads = value_and_grad(_loss_fun, argnums=0)(
                net_params, input_data, target_data
            )

            opt_update_state = self.opt_update(step_i, grads, opt_state)

            return loss, opt_update_state

        # initial state of the optimizer
        opt_state = self.opt_state_init

        # run in batches to train the model
        num_cosmo = input_data.shape[0]
        if batch_size > num_cosmo:
            msg = (
                f"Batch size {batch_size} is larger than size of data {num_cosmo}.\n"
                f"Setting `batch_size={num_cosmo}` instead, to not duplicate data."
            )
            warnings.warn(msg, UserWarning)
            batch_size = num_cosmo

        time_start = time()
        loss_history = []
        ran_key = jran.PRNGKey(seed)
        for i in range(num_epochs):
            ran_key, sk_idx = jran.split(ran_key, 2)
            idx = jran.randint(sk_idx, (batch_size,), 0, num_cosmo)
            x_train = input_data[idx]
            targets = target_data[idx]

            loss, opt_state = _train_step(i, opt_state, x_train, targets)
            loss_history.append(float(loss))

        time_end = time()
        self.train_dt_sec = time_end - time_start
        self.train_dt = format_time(time_end - time_start)
        if timeit:
            print("Training completed after %s" % self.train_dt)

        # training results
        self.opt_state_final = opt_state
        self.loss_history = loss_history

        return loss_history, opt_state

    def save_model(
        self,
        savedir=PRETRAINED_PATH,
        save_base_name=DEFAULT_MLP_MODEL,
        verbose=False,
    ):
        """
        Convenient function to save the model to disk

        Parameters
        ----------
        savedir: str
            path to directory where model is stored

        save_base_name: str
            base name for generated files

        verbose: bool
            if True, feedback will be printed

        Returns
        -------
        saved files at the requested path
        """
        savedir = os.path.join(savedir, save_base_name)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if savedir[-1] != "/":
            savedir += "/"

        # save final state
        file_path = savedir + save_base_name + ".pkl"
        with open(file_path, "wb") as file:
            pickle.dump(self.opt_state_final, file)

        # save loss history
        file_path = savedir + save_base_name + "_loss_hist" + ".npy"
        np.save(file_path, np.asarray(self.loss_history))

        # save additional information
        n_neuron_per_layer = self.get_mlp_architecture()[0]
        n_mlp_params = self._get_number_of_mlp_params(
            n_neuron_per_layer,
        )

        _layers = "{}".format(n_neuron_per_layer[0])
        for _layer in n_neuron_per_layer[1:]:
            _layers += ",{}".format(_layer)

        # ! Note: (_info_train_dt,...,_info_mlp_layers) must be first
        #         and rows up to, and including, _info_mlp_layers
        #         must be skipped when loading the model from files
        _info_train_dt = np.array(["train time:", self.train_dt])
        _info_mlp_n_params = np.array(["number of model parameters:", n_mlp_params])
        _info_mlp_n_layers = np.array(["number of layers:", len(_layers)])
        _info_mlp_layers = np.array(["neurons per layer:", _layers])
        _info_train_dt_sec = np.array(["train time (sec):", self.train_dt_sec])
        _info_batch_size = np.array(["batch size:", str(self.batch_size)])
        _info_num_batches = np.array(["n_batches:", str(self.num_batches)])
        _info_num_epochs = np.array(["n_epochs:", str(self.num_epochs)])
        _info_step_size = np.array(["step size:", "%.3e" % self.step_size])
        _info_opt_seed = np.array(["optimizer seed:", str(self.opt_seed)])
        additional_info = np.column_stack(
            (
                _info_train_dt,
                _info_mlp_n_params,
                _info_mlp_n_layers,
                _info_mlp_layers,
                _info_train_dt_sec,
                _info_batch_size,
                _info_num_batches,
                _info_num_epochs,
                _info_step_size,
                _info_opt_seed,
            )
        ).T
        np.savetxt(
            savedir + save_base_name + "_info.txt",
            additional_info,
            fmt="%s",
        )

        if verbose:
            print("Model and data saved in %s" % savedir)

    @classmethod
    def load_model(
        cls,
        savedir=PRETRAINED_PATH,
        save_base_name=DEFAULT_MLP_MODEL,
        verbose=False,
        load_extra=False,
    ):
        """
        Convenient function to load the model from disk

        Parameters
        ----------
        savedir: str
            path to directory where model is stored

        save_base_name: str
            base name for generated files

        verbose: bool
            if True, feedback will be printed

        load_extra: bool
            if True will load extra information from training,
            which is not needed for making a model prediction

        Returns
        -------
        model_final: namedtuple
            state of model at the optimizer's final step

        model_init: namedtuple
            initial state of the model

        loss_hist: ndarray of shape (num_steps,)
            loss at each step of the optimization
        """
        savedir = os.path.join(savedir, save_base_name)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if savedir[-1] != "/":
            savedir += "/"

        self = cls()

        # load final state
        file_path = savedir + save_base_name + ".pkl"
        with open(file_path, "rb") as file:
            self.opt_state_final = pickle.load(file)

        if verbose:
            print("Model and data loaded from %s" % savedir)

        # load additional information
        (
            train_dt_sec,
            batch_size,
            num_batches,
            num_epochs,
            step_size,
            opt_seed,
        ) = np.loadtxt(
            savedir + save_base_name + "_info.txt",
            usecols=(-1,),
            skiprows=4,
        )
        self.train_dt_sec = float(train_dt_sec)
        self.train_dt = format_time(self.train_dt_sec)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.num_epochs = int(num_epochs)
        self.step_size = float(step_size)
        self.opt_seed = int(opt_seed)

        (
            self.n_neuron_per_layer,
            self.n_layer,
            self.n_feature,
            self.n_target,
            self.n_mlp_params,
        ) = self.get_mlp_architecture(self.opt_state_final)

        if load_extra:
            # load loss history
            file_path = savedir + save_base_name + "_loss_hist" + ".npy"
            self.loss_hist = np.load(file_path)

        (
            self.opt_init,
            self.opt_update,
            self.get_params,
        ) = jax_opt.adam(self.step_size)

        self.hidden_layers = self.n_neuron_per_layer[1:-1]

        self.net_init, self.net_apply = self.init_mlp()

        self.load_successful = True

        return self

    def get_available_models(self):
        return PRETRAINED_MODEL_NAMES

    def get_mlp_architecture(
        self,
        state=None,
        verbose=False,
    ):
        """
        Helper function to get the architecture
        of the mlp from its state

        Parameters
        ----------
        state: nametuple
            state of the MLP from which to extract the information

        verbose: bool
            if True, information about the model will be printed

        Returns
        -------
        n_neuron_per_layer: list
            number of neurons per layer

        n_layer: int
            number of total layers

        n_feature: int
            number of features

        n_target: int
            number of targets
        """
        if state is None:
            state = self.opt_state_final

        n_neuron_per_layer = []
        shape_list = []
        for i in range(0, len(state.packed_state), 2):
            _shape = np.shape(state.packed_state[i][0])
            n_neuron_per_layer.append(_shape[1])
            shape_list.append(_shape)

        self.n_feature = shape_list[0][0]
        self.n_target = shape_list[-1][1]
        self.n_neuron_per_layer = [self.n_feature] + n_neuron_per_layer
        self.n_layer = len(n_neuron_per_layer)
        self.n_mlp_params = self._get_number_of_mlp_params(
            self.n_neuron_per_layer,
        )

        if verbose:
            print("Number of layers: %d" % self.n_layer)
            print("Neurons per layer: ", self.n_neuron_per_layer)
            print("Number of features: %d" % self.n_feature)
            print("Number of targets: %d" % self.n_target)
            print("Number of model parameters: %d" % self.n_mlp_params)

        return (
            self.n_neuron_per_layer,
            self.n_layer,
            self.n_feature,
            self.n_target,
            self.n_mlp_params,
        )

    def _get_number_of_mlp_params(self, n_neuron_per_layer):
        """
        Helper function to compute the total
        number of parameters of the model
        """
        hidden_layers = np.asarray(n_neuron_per_layer[1:-1])
        n_feature = n_neuron_per_layer[0]
        n_target = n_neuron_per_layer[-1]
        n_mlp_params = n_feature * hidden_layers[0]
        n_mlp_params += np.sum(hidden_layers[1:] * hidden_layers[:-1])
        n_mlp_params += hidden_layers[-1] * n_target
        n_mlp_params += np.sum(hidden_layers)
        n_mlp_params += n_target

        return n_mlp_params

    def reset(self):
        self.loss_function = None
        self.loss_args = None
        self.opt_state_init = None
        self.opt_state_final = None
        self.loss_history = None
        self.net_init = None
        self.net_apply = None
        self.net_params_init = None
        self.n_neuron_per_layer = None
        self.n_layer = None
        self.n_feature = None
        self.n_target = None
        self.batch_size = None
        self.num_batches = None
        self.num_epochs = None
        self.train_dt = None
        self.train_dt_sec = None
        self.step_size = None
        self.opt_seed = None
        self.n_mlp_params = None
        self.n_targets = None
        self.hidden_layers = None
        self.load_successful = None
