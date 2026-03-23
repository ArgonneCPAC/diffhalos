"""HMF parameter utils"""

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp

from ..utils.sigmoid_utils import _sigmoid
from ..calibrations.hmf_cal import DEFAULT_HMF_PARAMS
from ..calibrations.hmf_cal.smdpl_hmf import (
    HMF_Params,
    Ytp_Params,
    X0_Params,
    Lo_Params,
    Hi_Params,
)


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

N_DIFFSKY_HMF_PARAMS = 19

# x0, k, ymin, ymax
SIGMOID_PARAMS = (0.0, 0.13, -15.0, 15.0)


__all__ = (
    "define_diffsky_hmf_params_namedtuple_from_array",
    "define_diffsky_hmf_params_array_from_namedtuple",
)


@jjit
def get_bounded_params(u_params, sigmoid_params=SIGMOID_PARAMS):
    """
    Get the bounded version of the parameters

    Parameters
    ----------
    u_params: ndarray of shape (n_params, )
        unbounded parameters

    sigmoid_params: tuple
        sigmoid parameters for bounding

    Returns
    -------
    params: ndarray of shape (n_params, )
        bounded parameters
    """
    params = jnp.zeros(N_DIFFSKY_HMF_PARAMS)
    for i in range(N_DIFFSKY_HMF_PARAMS):
        params = params.at[i].set(_sigmoid(u_params[i], *sigmoid_params))

    return params


@jjit
def define_diffsky_hmf_params_namedtuple_from_array(params):
    """
    Helper function to define a diffksy HMF param
    namedtuple from a flat array

    Parameters
    ----------
    params: ndarray of shape (n_hmf_params, )
        diffsky hmf parameters as a flat array

    Returns
    -------
    hmf_ntup: namedtuple
        HMF parameters for diffsky model predictions
    """
    ytp_ntup = Ytp_Params(*params[0:5])
    x0_ntup = X0_Params(*params[5:10])
    lo_ntup = Lo_Params(*params[10:14])
    hi_ntup = Hi_Params(*params[14:19])
    hmf_args = (ytp_ntup, x0_ntup, lo_ntup, hi_ntup)
    hmf_ntup = HMF_Params(*hmf_args)

    return hmf_ntup


@jjit
def define_diffsky_hmf_params_array_from_namedtuple(params):
    """
    Helper function to define a diffksy HMF param
    array from a namedtuple

    Parameters
    ----------
    params: namedtuple
        diffsky hmf namedtuple

    Returns
    -------
    hmf_array: ndarray of shape (n_hmf_params, )
        HMF parameters flat array
    """
    hmf_array = np.zeros(N_DIFFSKY_HMF_PARAMS)
    icur = 0
    for _ntup in DEFAULT_HMF_PARAMS:
        npar = len(_ntup)
        hmf_array[icur : icur + npar] = np.asarray(_ntup)
        icur += npar

    return hmf_array
