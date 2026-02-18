"""HMF parameter utils"""

import numpy as np
import typing
from collections import namedtuple


from ..calibrations.hmf_cal import DEFAULT_HMF_PARAMS

__all__ = ("get_hmf_param_names", "define_diffsky_hmf_params_namedtuple")


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

YTP_PARAMS_ARRAY_DEFAULT = np.asarray(DEFAULT_HMF_PARAMS.ytp_params)
X0_PARAMS_ARRAY_DEFAULT = np.asarray(DEFAULT_HMF_PARAMS.x0_params)
LO_PARAMS_ARRAY_DEFAULT = np.asarray(DEFAULT_HMF_PARAMS.lo_params)
HI_PARAMS_ARRAY_DEFAULT = np.asarray(DEFAULT_HMF_PARAMS.hi_params)
HMF_PARAMS_ARRAY_DEFAULT = np.concatenate(
    [
        YTP_PARAMS_ARRAY_DEFAULT,
        X0_PARAMS_ARRAY_DEFAULT,
        LO_PARAMS_ARRAY_DEFAULT,
        HI_PARAMS_ARRAY_DEFAULT,
    ]
)


class Ytp_Params(typing.NamedTuple):
    """
    Structure of diffsky Ytp parameters namedtuple
    """

    ytp_ytp: float
    ytp_x0: float
    ytp_k: float
    ytp_ylo: float
    ytp_yhi: float


class X0_Params(typing.NamedTuple):
    """
    Structure of diffsky X0 parameters namedtuple
    """

    x0_ytp: float
    x0_x0: float
    x0_k: float
    x0_ylo: float
    x0_yhi: float


class Lo_Params(typing.NamedTuple):
    """
    Structure of diffsky Lo parameters namedtuple
    """

    lo_x0: float
    lo_k: float
    lo_ylo: float
    lo_yhi: float


class HI_Params(typing.NamedTuple):
    """
    Structure of diffsky parameters namedtuple
    """

    hi_ytp: float
    hi_x0: float
    hi_k: float
    hi_ylo: float
    hi_yhi: float


class HMF_Params(typing.NamedTuple):
    """
    Structure of diffsky parameters as namedtuple
    """

    ytp_params: Ytp_Params
    x0_params: X0_Params
    lo_params: Lo_Params
    hi_params: HI_Params


class HMF_Params_Flat(typing.NamedTuple):
    """
    Structure of diffsky parameters in namedtuple
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
    params=HMF_PARAMS_ARRAY_DEFAULT,
    hmf_param_names=HMF_PARAM_FIELDS_DIFFSKY,
    ytp_param_names=YTP_PARAM_FIELDS_DIFFSKY,
    x0_param_names=X0_PARAM_FIELDS_DIFFSKY,
    lo_param_names=LO_PARAM_FIELDS_DIFFSKY,
    hi_param_names=HI_PARAM_FIELDS_DIFFSKY,
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

    ytp_param_names: tuple
        hmf parameter ``ytp`` names like in ``Ytp_Params``

    x0_param_names: tuple
        hmf parameter ``x0`` names like in ``X0_Params``

    lo_param_names: tuple
        hmf parameter ``lo`` names like in ``Lo_Params``

    hi_param_names: tuple
        hmf parameter ``hi`` names like in ``Hi_Params``

    Returns
    -------
    hmf_ntup: namedtuple
        HMF parameters for diffsky model predictions
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
