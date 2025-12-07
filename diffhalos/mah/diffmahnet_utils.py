"""
Useful diffmahnet functions
See https://diffmahnet.readthedocs.io/en/latest/installation.html
"""

import numpy as np
import os
import pathlib
import glob

import jax
import jax.numpy as jnp

from diffmah import mah_halopop
from diffmah import DEFAULT_MAH_PARAMS
from diffmah.diffmah_kernels import DEFAULT_MAH_U_PARAMS
from diffmah.diffmah_kernels import (
    get_unbounded_mah_params,
    get_bounded_mah_params,
)

import diffmahnet

from .utils import rescale_mah_parameters

DEFAULT_MAH_UPARAMS = get_unbounded_mah_params(DEFAULT_MAH_PARAMS)

T_GRID_MIN = 0.5
T_GRID_MAX = jnp.log10(13.8)
N_T_GRID = 100

__all__ = (
    "mc_mah_cenpop",
    "get_mean_and_std_of_mah",
    "get_mah_from_unbounded_params",
    "load_diffmahnet_training_data",
    "get_available_models",
)


def mc_mah_cenpop(
    m_obs,
    t_obs,
    randkey,
    n_sample=1,
    centrals_model_key="cenflow_v2_0.eqx",
    t_min=T_GRID_MIN,
    t_max=T_GRID_MAX,
    n_t=N_T_GRID,
    return_mah_params=False,
):
    """
    Generate MC realiations of central halo populations
    using the ``diffmahnet`` code for ``diffmahpop``.
    This function takes in a grid of halo mass and
    cosmic time at observation and for each it generates
    MC samples given random keys

    Parameters
    ----------
    m_obs: ndarray of shape (n_cens, )
        grid of base-10 log of mass of the halos at observation, in Msun

    t_obs: ndarray of shape (n_cens, )
        grid of base-10 log of cosmic time at observation of each halo, in Gyr

    randkey: key
        JAX random key

    n_sample: int
        number of MC samples per (m_obs, t_obs) pair

    centrals_model_key: str
        model name for centrals

    t_min: float
        base-10 log of minimum value for time grid
        at which to compute mah, in Gyr

    t_max: float
        base-10 log of maximum value for time grid,
        at which to compute mah, in Gyr

    n_t: int
        number of points in time grid

    return_mah_params: bool
        if True the MAH parameters from the normalizing flow
        will also be returned

    Returns
    -------
    cen_mah: ndarray of shape (n_sample*n_cens, n_t)
        base-10 log of halo mass assembly histories,
        for all MC realizations, in Msun

    t_grid: ndarray of shape (n_sample*n_cens, n_t)
        cosmic time grid on which to compute MAHs,
        for all MC realizations, in Gyr

    cenflow_diffmahparams: namedtuple
        diffmah parameters from normalizing flow,
        each parameter is a ndarray of shape(n_sample*n_cens, )
    """
    # create diffmahnet model for centrals
    centrals_model = diffmahnet.load_pretrained_model(centrals_model_key)
    mc_diffmahnet_cenpop = centrals_model.make_mc_diffmahnet()

    # get a list of (m_obs, t_obs) for each MC realization
    m_vals, t_vals = [
        jnp.repeat(x.flatten(), n_sample)
        for x in np.stack(
            [m_obs, t_obs],
            axis=-1,
        ).T
    ]

    # get diffmah parameters from the normalizing flow
    keys = jax.random.split(randkey, 2)
    cenflow_diffmahparams = mc_diffmahnet_cenpop(
        centrals_model.get_params(), m_vals, t_vals, keys[0]
    )

    # construct time grids for each halo, given observation time
    t_grid = jnp.linspace(t_min, t_vals, n_t).T

    # compute the uncorrected predicted observed halo masses
    logm_obs_uncorrected = diffmahnet.log_mah_kern(
        cenflow_diffmahparams,
        t_grid,
        t_max,
    )[:, -1]

    # rescale the mah parameters to the correct logm0
    cenflow_diffmahparams = rescale_mah_parameters(
        cenflow_diffmahparams,
        m_vals,
        logm_obs_uncorrected,
    )

    # compute mah with corrected parameters
    cen_mah = diffmahnet.log_mah_kern(
        cenflow_diffmahparams,
        t_grid,
        t_max,
    )

    if return_mah_params:
        return cen_mah, t_grid, cenflow_diffmahparams

    return cen_mah, t_grid


def get_mean_and_std_of_mah(mah):
    """
    Helper function to get the mean and 1-sigma
    standard deviation of a sample of mah realizations

    Parameters
    ----------
    mah: ndarray of shape (n_halo, n_t)
        MAH of the population of halos

    Returns
    -------
    mah_mean: ndarray of shape (n_halo, )
        mean of mah at each time

    mah_max: ndarray of shape (n_halo, )
        upper bound for 1-sigma band around mean

    mah_min: ndarray of shape (n_halo, )
        lower bound for 1-sigma band around mean
    """
    n_t = mah.shape[1]

    mah_mean = np.zeros(n_t)
    mah_max = np.zeros(n_t)
    mah_min = np.zeros(n_t)
    for t in range(n_t):
        _mah = mah[:, t]
        mah_mean[t] = np.mean(_mah)
        _std = np.std(_mah)
        mah_max[t] = mah_mean[t] + _std
        mah_min[t] = mah_mean[t] - _std

    return mah_mean, mah_max, mah_min


def get_mah_from_unbounded_params(
    mah_params_unbound,
    logt0,
    t_grid,
    logm_obs,
):
    """
    Helper function to generate the MAH from
    a set of diffmah unbounded parameters,
    for a population of halos

    Parameters
    ----------
    mah_params_unbound: ndarray of shape (n_halo, n_mah_param)
        unbounded ``diffmah`` parameters
        (logm0, logtc, early_index, late_index, t_peak)

    logt0: float
        base-10 log of the age of the Universe at z=0, in Gyr

    t_grid: ndarray of shape (n_t, )
        cosmic time grid at which to compute the MAH

    logm_obs: float
        base-10 log of observed halo mass, in Msun

    Returns
    -------
    log_mah: ndarray of shape (n_halo, n_t)
        base-10 log of MAH, in Msun
    """
    mah_params_bound = jnp.array(
        [
            *get_bounded_mah_params(
                DEFAULT_MAH_U_PARAMS._make(mah_params_unbound.T),
            )
        ]
    )

    mah_params_uncorrected = DEFAULT_MAH_PARAMS._make(mah_params_bound)

    _, logm_obs_uncorrected = mah_halopop(
        mah_params_uncorrected,
        t_grid,
        logt0,
    )

    # rescale the mah parameters to the correct logm0
    mah_params = rescale_mah_parameters(
        mah_params_uncorrected,
        logm_obs,
        logm_obs_uncorrected[:, -1],
    )

    _, log_mah = mah_halopop(mah_params, t_grid, logt0)

    return log_mah


def load_diffmahnet_training_data(
    path=None,
    is_test: bool | str = False,
    is_cens=True,
):
    """
    Convenient function to load the data
    used to train ``diffmahnet``

    Parameters
    ----------
    path: str
        path to the training data folder;
        is not provided directly, the environment variable
        ``DIFFMAHNET_TRAINING_DATA`` will be used instead

    is_test: bool or str
        slices the training data into smaller test data

    is_cens: bool
        if True, data for centrals will be loaded,
        if False, data for satellites will be loaded

    Returns
    -------
    x_unbound: ndarray of shape (n_pdf_var, )
        PDF variables

    u: ndarray of shape (n_cond_var, )
        conditional variables
    """
    if path is None:
        try:
            path = os.environ["DIFFMAHNET_TRAINING_DATA"]
        except KeyError:
            msg = (
                "Since you did not pass the 'filename' argument\n"
                "then you must have the 'DIFFMAHNET_TRAINING_DATA' environment variable set.\n"
                "Run first 'export ``DIFFMAHNET_TRAINING_DATA=path_to_data_folder``'"
            )
            raise ValueError(msg)

    # Parse available training data files
    tdata_files = glob.glob(str(pathlib.Path(path) / "*"))
    filenames = [x.split("/")[-1] for x in tdata_files]
    lgm_vals = np.array([float(x.split("_")[1]) for x in filenames])
    t_vals = np.array([float(x.split("_")[3]) for x in filenames])
    is_cens_vals = np.array([x.split(".")[-2] == "cens" for x in filenames])
    fileinfo = list(
        zip(
            tdata_files,
            lgm_vals.tolist(),
            t_vals.tolist(),
            is_cens_vals.tolist(),
        )
    )
    cen_file_inds = np.where(is_cens_vals)[0]
    sat_file_inds = np.where(~is_cens_vals)[0]

    # Load data
    test_train_file_split = 80  # about 25:75 test-train split ratio
    if is_test == "both":
        test_train_file_split = None
    inds = cen_file_inds if is_cens else sat_file_inds
    test_train_slice = slice(None, test_train_file_split)
    if is_test:
        test_train_slice = slice(test_train_file_split, None)
    inds = inds[test_train_slice]

    x = []  # PDF variables
    u = []  # conditional variables
    for i in inds:
        filename, lgm, t, is_cens_val = fileinfo[i]
        assert is_cens == is_cens_val
        x.append(np.load(filename))
        u.append(np.tile(np.array([[lgm, t]]), (x[-1].shape[0], 1)))

    x = jnp.concatenate(x, axis=0)
    u = jnp.concatenate(u, axis=0)

    # Transfrorm x parameters from bounded to unbounded space
    x_unbound = jnp.array(
        [
            *get_unbounded_mah_params(
                DEFAULT_MAH_PARAMS._make(x.T),
            )
        ]
    ).T

    isfinite = np.all((jnp.isfinite(x_unbound)), axis=1)
    return x_unbound[isfinite], u[isfinite]


def get_available_models():
    available_names = diffmahnet.pretrained_model_names
    print(available_names)
