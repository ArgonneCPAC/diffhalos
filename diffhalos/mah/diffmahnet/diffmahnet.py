import json
import glob
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import flowjax

import flowjax.distributions
import flowjax.flows
import flowjax.train
import equinox as eqx
import paramax

try:
    from diffopt import kdescent

    HAS_DIFFOPT = True
except ImportError:
    HAS_DIFFOPT = False

import diffmah
from diffmah import DEFAULT_MAH_PARAMS
from diffmah.diffmah_kernels import get_unbounded_mah_params, get_bounded_mah_params

DEFAULT_MAH_UPARAMS = get_unbounded_mah_params(DEFAULT_MAH_PARAMS)


log_mah_kern = jax.jit(
    jax.vmap(diffmah.diffmah_kernels._log_mah_kern, in_axes=(0, 0, None))
)

pretrained_path = pathlib.Path(__file__).parent / "pretrained_models"
pretrained_model_names = glob.glob(str(pretrained_path / "*.eqx"))
pretrained_model_names = [str(pathlib.Path(x).name) for x in pretrained_model_names]


def load_pretrained_model(name):
    """
    Load a pretrained model from the diffmahnet package

    Parameters
    ----------
    name : str
        Name of the model to load. Should be one of the following:
        "diffmahnet_1", "diffmahnet_2", "diffmahnet_3"

    Returns
    -------
    DiffMahFlow
        The loaded model
    """
    if name not in pretrained_model_names:
        raise ValueError(
            f"{name=} not found. Available models: {pretrained_model_names}"
        )

    filename = pretrained_path / name
    return DiffMahFlow.load(filename)


def gen_time_grids(key, t_obs, t_min=0.1, n_tgrid=20):
    key1, key2 = jax.random.split(key, 2)
    fixed_tgrid = jnp.linspace(t_min, t_obs, n_tgrid).T
    t_min_dither = jax.random.uniform(
        key1, t_obs.shape, minval=fixed_tgrid[:, 0], maxval=fixed_tgrid[:, 1]
    )
    t_max_dither = jax.random.uniform(
        key2, t_obs.shape, minval=fixed_tgrid[:, -2], maxval=fixed_tgrid[:, -1]
    )
    return np.linspace(t_min_dither, t_max_dither, n_tgrid).T


def scaler_transform(x, scaler, inverse=False):
    mean, scale = scaler.mean_, scaler.scale_
    if inverse:
        return x * scale + mean
    return (x - mean) / scale


def make_flatten_and_unflatten_funcs(param_tree):
    flatparams, treedef = jax.tree.flatten(param_tree)
    sizes = [xi.size for xi in flatparams]
    shapes = [xi.shape for xi in flatparams]

    @jax.jit
    def flatten(x):
        return jnp.concat([jnp.ravel(xi) for xi in jax.tree.flatten(x)[0]])

    @jax.jit
    def unflatten(flatx):
        x = []
        index = 0
        for shape, size in zip(shapes, sizes):
            xi = flatx[index : (index := index + size)]
            x.append(jnp.reshape(xi, shape))
        return jax.tree.unflatten(treedef, x)

    return flatten, unflatten


class DiffMahFlow:
    """
    The primary class within diffmahnet. This class is used to train and
    from and emulate the diffmahpop model.

    Parameters
    ----------
    scaler : Scaler
        Scaler object to normalize the input data
    nn_depth : int, optional
        Depth of the neural network, by default 2
    nn_width : int, optional
        Number of hidden layers in the neural network, by default 50
    flow_layers : int, optional
        Number of flow layers, by default 8
    randkey : jax.random.PRNGKey, optional
        Random key for reproducibility, by default None
    """

    def __init__(self, scaler, nn_depth=2, nn_width=50, flow_layers=8, randkey=None):
        x_dim = scaler.x_scaler.n_features_in_
        cond_dim = scaler.u_scaler.n_features_in_
        self.scaler = scaler
        self.randkey = jax.random.key(0) if randkey is None else randkey
        self.flow = flowjax.flows.masked_autoregressive_flow(
            key=self.randkey,
            invert=False,
            base_dist=flowjax.distributions.Normal(jnp.zeros(x_dim)),
            cond_dim=cond_dim,
            nn_depth=nn_depth,
            nn_width=nn_width,
            flow_layers=flow_layers,
        )
        self.nn_depth = nn_depth
        self.nn_width = nn_width
        self.flow_layers = flow_layers
        param_tree, self.static = self._partition()
        self.flatten, self.unflatten = make_flatten_and_unflatten_funcs(param_tree)

    def get_tgrid_and_log_mah(
        self, m_obs, t_obs, randkey, t_min=0.1, n_tgrid=20, t0=13.8, extra_shape=()
    ):
        key1, key2 = jax.random.split(randkey)
        u = jnp.array([m_obs, t_obs]).T
        mah_params = self.sample(
            u, randkey=key1, extra_shape=extra_shape, asparams=True
        )

        tgrid = gen_time_grids(key2, t_obs, t_min=t_min, n_tgrid=n_tgrid)
        log_mah = log_mah_kern(mah_params, tgrid, np.log10(t0))
        return tgrid, log_mah

    def make_mc_diffmahnet(self):
        @jax.jit
        def mc_diffmahnet(flow_params, lgm_obs, t_obs, ran_key):
            return self.sample(
                jnp.array([lgm_obs, t_obs]).T,
                randkey=ran_key,
                asparams=True,
                flow_params=flow_params,
            )

        return mc_diffmahnet

    def sample(
        self, condition, randkey=None, extra_shape=(), asparams=False, flow_params=None
    ):
        """
        Sample diffmah u_params, conditioned on (m_obs, t_obs)

        Parameters
        ----------
        condition : jnp.ndarray
            Array of m_obs and t_obs, of shape (n_samples, 2)
        randkey : jax.random.PRNGKey, optional
            Random key for reproducibility
        extra_shape : tuple, optional
            Extra shape to repeatedly sample for each condition value
        asparams : bool, optional
            If true, return DiffmahParams tuple instead of uparams array
        flow_params: jnp.ndarray, optional
            Set the parameters of the flow to this value for sampling
            (for functional programming instead of object-oriented)

        Returns
        -------
        jnp.ndarray | DiffmahParams
            Sampled unbound params, of shape (n_samples, 5, *extra_shape)
        """
        if flow_params is not None:
            flow = self._flow_from_flat_params(flow_params)
        else:
            flow = self.flow
        condition_scaled = scaler_transform(condition, self.scaler.u_scaler)
        if randkey is None:
            randkey, self.randkey = jax.random.split(self.randkey, 2)
        x_scaled = flow.sample(randkey, extra_shape, condition=condition_scaled)
        uparam_array = scaler_transform(x_scaled, self.scaler.x_scaler, inverse=True)
        if asparams:
            return get_bounded_mah_params(DEFAULT_MAH_UPARAMS._make(uparam_array.T))
        else:
            return uparam_array

    def get_params(self):
        param_tree = self._partition()[0]
        return self.flatten(param_tree)

    def set_params(self, flat_params):
        self.flow = self._flow_from_flat_params(flat_params)
        self._reset_static()

    def save(self, filename):
        """
        Save this model object to an eqx file for future use

        Parameters
        ----------
        filename : str
            Filename to save the model to. The ".eqx" extension will be added
            automatically if not present
        """
        hyperparams = dict(
            nn_depth=self.nn_depth,
            nn_width=self.nn_width,
            flow_layers=self.flow_layers,
            **self.scaler.to_dict(),
        )

        filename = str(filename).removesuffix(".eqx") + ".eqx"
        with open(filename, "wb") as f:
            f.write((json.dumps(hyperparams) + "\n").encode())
            eqx.tree_serialise_leaves(f, self.flow)

    @classmethod
    def load(cls, filename, randkey=None):
        """
        Load a pre-trained model from an eqx file

        Parameters
        ----------
        filename : str
            Filename to load the model from. The ".eqx" extension will be
            added automatically if not present
        randkey : jax.random.PRNGKey, optional
            Random key for reproducibility

        Returns
        -------
        DiffMahFlow
            The loaded model
        """
        randkey = jax.random.key(0) if randkey is None else randkey
        filename = str(filename).removesuffix(".eqx") + ".eqx"
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            hyperparams["nn_depth"] = int(hyperparams["nn_depth"])
            hyperparams["nn_width"] = int(hyperparams["nn_width"])
            hyperparams["flow_layers"] = int(hyperparams["flow_layers"])
            scaler = Scaler.from_dict(hyperparams)
            self = cls(
                scaler=scaler,
                nn_depth=hyperparams["nn_depth"],
                nn_width=hyperparams["nn_width"],
                flow_layers=hyperparams["flow_layers"],
                randkey=randkey,
            )
            self.flow = eqx.tree_deserialise_leaves(f, self.flow)

        self._reset_static()
        return self

    def init_fit(
        self,
        xtrain,
        utrain,
        randkey=None,
        learning_rate=1e-2,
        max_patience=10,
        max_epochs=50,
    ):
        """Train the flow directly on P(mah_params|m_obs,t_obs)"""
        x_scaled = scaler_transform(xtrain, self.scaler.x_scaler)
        u_scaled = scaler_transform(utrain, self.scaler.u_scaler)
        if randkey is None:
            randkey, self.randkey = jax.random.split(self.randkey, 2)
        self.flow, losses = flowjax.train.fit_to_data(
            randkey,
            self.flow,
            x_scaled,
            condition=u_scaled,
            learning_rate=learning_rate,
            max_patience=max_patience,
            max_epochs=max_epochs,
        )
        self._reset_static()
        return losses

    def adam_fit(
        self,
        lossfunc,
        randkey=None,
        nsteps=100,
        progress=True,
        learning_rate=1e-4,
        thin=1,
        **kwargs,
    ):
        """Fit the flow using the Adam stochastic gradient descent

        Parameters
        ----------
        lossfunc : callable
            Loss function to minimize, should have signature
            `lossfunc(diffmahflow, randkey=key) -> float`
        randkey : PRNG Key, optional
            Set the random seed, by default use the current self.randkey
        nsteps : int, optional
            Number of Adam steps to perform, by default 100
        progress : bool, optional
            Set false to hide progress bars, by default True
        learning_rate : float, optional
            Initial Adam learning rate, by default 1e-4
        thin : int, optional
            Return parameters for every `thin` iterations, by default 1. Set
            `thin=0` to only return final parameters

        Returns
        -------
        jnp.array[float]
            Loss value at each step of the descent
        """
        if not HAS_DIFFOPT:
            msg = "Must have diffopt installed to use adam_fit"
            raise ImportError(msg)

        if randkey is None:
            randkey, self.randkey = jax.random.split(self.randkey, 2)

        @jax.jit
        def lossfunc_from_params(flat_params, randkey=randkey):
            self.set_params(flat_params)
            return lossfunc(self, randkey=randkey)

        adam_params, adam_losses = kdescent.adam(
            lossfunc_from_params,
            self.get_params(),
            nsteps=nsteps,
            progress=progress,
            randkey=randkey,
            learning_rate=learning_rate,
            thin=thin,
            **kwargs,
        )
        self.set_params(adam_params[-1])
        self.static = self._partition()[1]

        return adam_params, adam_losses

    def _partition(self):
        return eqx.partition(
            self.flow,
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )

    def _flow_from_flat_params(self, flat_params):
        param_tree = self.unflatten(flat_params)
        return eqx.combine(param_tree, self.static)

    def _reset_static(self):
        param_tree, self.static = self._partition()
        self.flatten, self.unflatten = make_flatten_and_unflatten_funcs(param_tree)


class _StandardScaler:
    def __init__(self, mean_=None, scale_=None):
        self.mean_ = mean_
        self.scale_ = scale_

    def fit(self, x):
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)

    @property
    def n_features_in_(self):
        return self.mean_.shape[0]


class Scaler:
    # Computes and stores scaling objects for X and U
    def __init__(self):
        self.x_scaler = _StandardScaler()
        self.u_scaler = _StandardScaler()

    @classmethod
    def compute(cls, x, u):
        self = cls()
        self.x_scaler.fit(x)
        self.u_scaler.fit(u)
        return self

    @classmethod
    def from_dict(cls, save_dict):
        self = cls()
        self.x_scaler.mean_ = np.array(save_dict["x_mean"])
        self.x_scaler.scale_ = np.array(save_dict["x_scale"])
        self.u_scaler.mean_ = np.array(save_dict["u_mean"])
        self.u_scaler.scale_ = np.array(save_dict["u_scale"])
        return self

    def to_dict(self):
        return dict(
            x_mean=self.x_scaler.mean_.tolist(),
            x_scale=self.x_scaler.scale_.tolist(),
            u_mean=self.u_scaler.mean_.tolist(),
            u_scale=self.u_scaler.scale_.tolist(),
        )
