"""HMF parameter utils"""

import jax
from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten

import numpy as np
import typing
from collections import namedtuple

__all__ = (
    "get_hmf_param_names",
    "define_diffsky_hmf_params_namedtuple",
    "diffsky_HMF_params_constructor",
)


HMF_PARAM_FIELDS_DIFFSKY = (
    "ytp_params",
    "x0_params",
    "lo_params",
    "hi_params",
)

HMF_FLAT_PARAM_FIELDS_DIFFSKY = (
    "ytp_ytp",
    "ytp_x0",
    "ytp_k",
    "ytp_ylo",
    "ytp_yhi",
    "x0_ytp",
    "x0_x0",
    "x0_k",
    "x0_ylo",
    "x0_yhi",
    "lo_x0",
    "lo_k",
    "lo_ylo",
    "lo_yhi",
    "hi_ytp",
    "hi_x0",
    "hi_k",
    "hi_ylo",
    "hi_yhi",
)

YTP_PARAM_FIELDS_DIFFSKY = (
    "ytp_ytp",
    "ytp_x0",
    "ytp_k",
    "ytp_ylo",
    "ytp_yhi",
)

X0_PARAM_FIELDS_DIFFSKY = (
    "x0_ytp",
    "x0_x0",
    "x0_k",
    "x0_ylo",
    "x0_yhi",
)

LO_PARAM_FIELDS_DIFFSKY = (
    "lo_x0",
    "lo_k",
    "lo_ylo",
    "lo_yhi",
)

HI_PARAM_FIELDS_DIFFSKY = (
    "hi_ytp",
    "hi_x0",
    "hi_k",
    "hi_ylo",
    "hi_yhi",
)


HMF_Params_diffsky = namedtuple("HMF_Params", HMF_PARAM_FIELDS_DIFFSKY)
HMF_Flat_Params_diffsky = namedtuple("HMF_Params", HMF_FLAT_PARAM_FIELDS_DIFFSKY)
YTP_Params_diffsky = namedtuple("Ytp_Params", YTP_PARAM_FIELDS_DIFFSKY)
X0_Params_diffsky = namedtuple("X0_Params", X0_PARAM_FIELDS_DIFFSKY)
LO_Params_diffsky = namedtuple("Lo_Params", LO_PARAM_FIELDS_DIFFSKY)
HI_Params_diffsky = namedtuple("Hi_Params", HI_PARAM_FIELDS_DIFFSKY)


class Ytp_Params(typing.NamedTuple):
    """
    Structure of diffsky parameters named tuple
    """

    ytp_ytp: float
    ytp_x0: float
    ytp_k: float
    ytp_ylo: float
    ytp_yhi: float


class X0_Params(typing.NamedTuple):
    """
    Structure of diffsky parameters named tuple
    """

    x0_ytp: float
    x0_x0: float
    x0_k: float
    x0_ylo: float
    x0_yhi: float


class Lo_Params(typing.NamedTuple):
    """
    Structure of diffsky parameters named tuple
    """

    lo_x0: float
    lo_k: float
    lo_ylo: float
    lo_yhi: float


class HI_Params(typing.NamedTuple):
    """
    Structure of diffsky parameters named tuple
    """

    hi_ytp: float
    hi_x0: float
    hi_k: float
    hi_ylo: float
    hi_yhi: float


class HMF_Params(typing.NamedTuple):
    """
    Structure of diffsky parameters named tuple
    """

    ytp_params: Ytp_Params
    x0_params: X0_Params
    lo_params: Lo_Params
    hi_params: HI_Params


class HMF_Params_Flat(typing.NamedTuple):
    """
    Structure of diffsky parameters named tuple
    """

    ytp_ytp: float
    ytp_x0: float
    ytp_k: float
    ytp_ylo: float
    ytp_yhi: float
    x0_ytp: float
    x0_x0: float
    x0_k: float
    x0_ylo: float
    x0_yhi: float
    lo_x0: float
    lo_k: float
    lo_ylo: float
    lo_yhi: float
    hi_ytp: float
    hi_x0: float
    hi_k: float
    hi_ylo: float
    hi_yhi: float


def get_hmf_param_names():
    """
    Helper function to return the naming structure
    of the Diffsky HMF model parameters
    """
    return (
        HMF_PARAM_FIELDS_DIFFSKY,
        YTP_PARAM_FIELDS_DIFFSKY,
        X0_PARAM_FIELDS_DIFFSKY,
        LO_PARAM_FIELDS_DIFFSKY,
        HI_PARAM_FIELDS_DIFFSKY,
    )


def define_diffsky_hmf_params_namedtuple(
    params,
    hmf_param_names=HMF_PARAM_FIELDS_DIFFSKY,
    ytp_param_names=YTP_PARAM_FIELDS_DIFFSKY,
    x0_param_names=X0_PARAM_FIELDS_DIFFSKY,
    lo_param_names=LO_PARAM_FIELDS_DIFFSKY,
    hi_param_names=HI_PARAM_FIELDS_DIFFSKY,
    ytp_params=None,
    x0_params=None,
    lo_params=None,
    hi_params=None,
):
    """
    Helper function to define a diffksy compatible
    named tuple holding HMF parameters for a
    model prediction

    Parameters
    ----------
    params: ndarray of shape (n_hmf_params,)
        all hmf parameter values concatenated,
        in the same order as the ``hmf_param_names``

    hmf_param_names: tuple
        hmf parameter names in same order as
        the values stored in ``params``

    ytp_params: ndarray of shape (n_ytp_params,)
        ``y_tp`` parameter values,
        if ``params`` not provided

    x0_params: ndarray of shape (n_x0_params,)
        ``x0`` parameter values,
        if ``params`` not provided

    lo_params: ndarray of shape (n_lo_params,)
        ``lo`` parameter values,
        if ``params`` not provided

    hi_params: ndarray of shape (n_hi_params,)
        ``hi`` parameter values,
        if ``params`` not provided

    Returns
    -------
    hmf_ntup: namedtuple
        HMF parameters for Diffsky model predictions
    """
    icur = 0
    for _param in hmf_param_names:
        if _param == "ytp_params":
            param_order = np.asarray(
                [ytp_param_names.index(_param) for _param in YTP_PARAM_FIELDS_DIFFSKY]
            )
            ytp_params = np.asarray(params[icur : icur + 5])[param_order]
            icur += 5
        elif _param == "x0_params":
            param_order = np.asarray(
                [x0_param_names.index(_param) for _param in X0_PARAM_FIELDS_DIFFSKY]
            )
            x0_params = np.asarray(params[icur : icur + 5])[param_order]
            icur += 5
        elif _param == "lo_params":
            param_order = np.asarray(
                [lo_param_names.index(_param) for _param in LO_PARAM_FIELDS_DIFFSKY]
            )
            lo_params = np.asarray(params[icur : icur + 4])[param_order]
            icur += 4
        elif _param == "hi_params":
            param_order = np.asarray(
                [hi_param_names.index(_param) for _param in HI_PARAM_FIELDS_DIFFSKY]
            )
            hi_params = np.asarray(params[icur : icur + 5])[param_order]
            icur += 5
        else:
            errmsg = "!ERROR! HMF parameter %s is invalid" % _param
            raise Exception(errmsg)

    # save the values of the parameters in the named tuples
    ytp_ntup = YTP_Params_diffsky(*ytp_params)
    x0_ntup = X0_Params_diffsky(*x0_params)
    lo_ntup = LO_Params_diffsky(*lo_params)
    hi_ntup = HI_Params_diffsky(*hi_params)
    hmf_args = (ytp_ntup, x0_ntup, lo_ntup, hi_ntup)
    hmf_ntup = HMF_Params_diffsky(*hmf_args)

    return hmf_ntup


class diffsky_HMF_params_constructor:
    def __init__(self):
        """
        Define an object to hold diffsky parameters
        both as a named tuple and an array,
        in order to able to transform from one to the other
        in a way that is jax-friendly and can be used with jit etc
        """
        self.pytree_struct = [
            Ytp_Params(*[1, 2, 3, 4, 5]),
            X0_Params(*[1, 2, 3, 4, 5]),
            Lo_Params(*[1, 2, 3, 4]),
            HI_Params(*[1, 2, 3, 4, 5]),
        ]
        _, self.schema = tree_flatten(self.pytree_struct)

    def array_to_ntup(self, params_array):
        """
        Convert an array of parameter values into a
        named tuple following the correct order of parameters
        """
        params_ntup = HMF_Params(*tree_unflatten(self.schema, params_array))
        return params_ntup

    def ntup_to_array(self, params_ntup):
        params_array, _ = tree_flatten(params_ntup)
        return params_array


def flatten_tuples(t):
    for x in t:
        if isinstance(x, tuple):
            yield from flatten_tuples(x)
        else:
            yield x


def tuple_to_jax_array(t):
    res = tuple(flatten_tuples(t))
    return jnp.asarray(res)


hmf_params_obj = diffsky_HMF_params_constructor()


def array_to_tuple(a, t):
    T = t(*hmf_params_obj.array_to_ntup(a))
    return T


def register_tuple(named_tuple_class):
    jax.tree_util.register_pytree_node(
        named_tuple_class,
        # tell JAX how to unpack the NamedTuple to an iterable
        lambda x: (tuple_to_jax_array(x), None),
        # tell JAX how to pack it back into the proper NamedTuple structure
        lambda _, x: array_to_tuple(x, named_tuple_class),
    )


# def flatten_func(obj):
#     """
#     Object that flattens the parameters
#     as needed to be used with jax
#     """
#     _children, aux_data = tree_flatten(obj.pytree_struct)
#     children = np.asarray(_children)

#     return (children, aux_data)


# def unflatten_func(aux_data, children):
#     """
#     Object that unflattens the parameters
#     as needed to be used with jax
#     """
#     obj = object.__new__(diffsky_HMF_params_constructor)
#     obj.params_array = children
#     (schema,) = aux_data
#     obj.params_ntup = tree_unflatten(schema, children)
#     return obj


# def jax_register_hmf_params_obj():
#     jax.tree_util.register_pytree_node(
#         diffsky_HMF_params_constructor,
#         flatten_func,
#         unflatten_func,
#     )
