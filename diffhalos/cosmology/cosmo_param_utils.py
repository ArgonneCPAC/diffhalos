"""Cosmological parameter utils"""

import numpy as np
from scipy.stats import qmc
from collections import namedtuple, OrderedDict

from jax import jit as jjit

from colossus.cosmology import cosmology

from .cosmo import (
    DEFAULT_COSMOLOGY,
    DEFAULT_COSMOLOGY_COLOSSUS,
    DEFAULT_COSMOLOGY_DSPS,
    DEFAULT_COSMO_PRIORS,
)

__all__ = (
    "sample_cosmo_params",
    "sample_cosmo_params_full_cosmo",
    "define_dsps_cosmology",
    "define_colossus_cosmology",
)


COSMO_PARAM_NAMES = DEFAULT_COSMOLOGY.keys()
COSMO_NTUP = namedtuple("CosmoNtup", COSMO_PARAM_NAMES)

AV_SAMPLING_METHODS = ("LatinHypercube",)


def sample_cosmo_params(
    cosmo_priors=DEFAULT_COSMO_PRIORS,
    seed=None,
    num_samples=1,
    method="LatinHypercube",
):
    """
    Sample cosmological parameters from prior volume
    following the requested scheme

    Parameters
    ----------
    cosmo_priors: dictionary
        cosmological parameter values

    seed: int
        randomly generated key

    num_samples: int
        number of cosmology samples to generate

    method: str
        method for sampling;
        options:
        - 'LatinHypercube': Latin Hypercube

    Returns
    -------
    cosmo_samples: ndarray of shape (num_samples, num_params)
        cosmological parameter samples
    """
    if method == "LatinHypercube":
        cosmo_samples = _sample_cosmo_params_latin_hypercube(
            cosmo_priors=cosmo_priors,
            seed=seed,
            num_samples=num_samples,
        )
    else:
        print("!ERROR! Method %s is not implemented" % method)
        errmsg = "Choose from: ('LatinHypercube')"
        raise Exception(errmsg)

    return cosmo_samples


def sample_cosmo_params_full_cosmo(
    underlying_cosmo=DEFAULT_COSMOLOGY,
    cosmo_priors=DEFAULT_COSMO_PRIORS,
    seed=None,
    num_samples=1,
    method="LatinHypercube",
):
    """
    Sample cosmological parameters from prior volume
    following the requested scheme

    Parameters
    ----------
    underlying_cosmo: dictionary
        underlying full cosmology

    cosmo_priors: dictionary
        cosmological parameter values

    seed: int
        randomly generated key

    num_samples: int
        number of cosmology samples to generate

    method: str
        method for sampling;
        options:
        - 'LatinHypercube': Latin Hypercube

    Returns
    -------
    cosmo_params: ndarray of shape (num_samples, num_cosmo_params)
        cosmological parameter samples as an array, where the
        order of the parameters per sample is the same as in `underlying_cosmo`
    """
    # all cosmological parameters
    cosmo_param_names = underlying_cosmo.keys()

    # sampled cosmological parameters
    cosmo_sampled_names = cosmo_priors.keys()

    assert np.all(
        [_p in cosmo_param_names for _p in cosmo_sampled_names]
    ), f"!ERROR! {cosmo_sampled_names} not all in {cosmo_param_names}"

    if method.lower() == "latinhypercube":
        cosmo_samples = _sample_cosmo_params_latin_hypercube(
            cosmo_priors=cosmo_priors,
            seed=seed,
            num_samples=num_samples,
        )
    else:
        errmsg = "!ERROR! Method {} is not implemented. \n Choose from: {}".format(
            method, AV_SAMPLING_METHODS
        )
        raise Exception(errmsg)

    cosmo_params = np.zeros((num_samples, len(cosmo_param_names)))
    for isam in range(num_samples):
        cntr = 0
        for iparam, _param in enumerate(cosmo_param_names):
            # parameter order in cosmo_samples same as in cosmo_sampled_names
            if _param in cosmo_sampled_names:
                cosmo_params[isam, iparam] = cosmo_samples[isam, cntr]
                cntr += 1
            else:
                cosmo_params[isam, iparam] = underlying_cosmo[_param]

    return cosmo_params


def _sample_cosmo_params_latin_hypercube(
    cosmo_priors=DEFAULT_COSMO_PRIORS,
    seed=None,
    num_samples=1,
):
    """
    Sample cosmological parameters from prior volume
    following a Latin Hypercube scheme

    Parameters
    ----------
    cosmo_priors: dictionary
        cosmological parameter values

    seed: int
        randomly generated key

    num_samples: int
        number of cosmology samples to generate

    Returns
    -------
    cosmo_samples: ndarray of shape (num_samples, num_params)
        cosmological parameter samples
    """
    # number of cosmological parameters to sample
    num_params = len(cosmo_priors)

    # Latin Hypercube samples (num_samples, num_params)
    LH = qmc.LatinHypercube(num_params, seed=seed)
    unit_hypercube = LH.random(n=num_samples)

    # rescale samples given the parameter priors
    cosmo_samples = np.zeros_like(unit_hypercube)
    for ip, par in enumerate(cosmo_priors.keys()):
        lbound, rbound = cosmo_priors[par]
        cosmo_samples[:, ip] = lbound + (rbound - lbound) * unit_hypercube[:, ip]

    return cosmo_samples


@jjit
def define_dsps_cosmology(cosmo_params_array):
    """
    Helper function to define a DSPS cosmology
    given an array of cosmological parameters

    Parameters
    ----------
    cosmo_params_array: ndarray of shape (n_cosmo_params, )
        array of cosmological parameters in the default form

    Returns
    -------
    cosmo_dsps: namedtuple
        dsps.cosmology with parameters matching the input array
    """
    cosmo_params_ntup = COSMO_NTUP(*cosmo_params_array)

    cosmo_dsps = DEFAULT_COSMOLOGY_DSPS
    cosmo_dsps = cosmo_dsps._replace(
        Om0=cosmo_params_ntup.Om0,
        h=cosmo_params_ntup.H0 / 100,
        w0=cosmo_params_ntup.w0,
        wa=cosmo_params_ntup.wa,
    )

    return cosmo_dsps


def define_colossus_cosmology(
    cosmo_name="ColossusCosmo",
    cosmo_params=DEFAULT_COSMOLOGY,
    flat=None,
    Om0=None,
    sigma8=None,
    ns=None,
    Ob0=None,
    H0=None,
    w0=None,
    wa=None,
):
    """
    Define Colossus cosmology

    Parameters
    ----------
    cosmo_name: str
        cosmology name

    cosmo_params: dictionary
        dictionary with all cosmological parameters
        needed to define a Colossus cosmology;
        default values are from Planck 2018 cosmology

    flat: bool
        if True, flat cosmology is assumed

    Om0: float
        value of Omega_matter

    sigma8: float
        value of sigma_8

    ns: float
        value of n_s

    Ob0: float
        value of Omega_baryon

    H0: float
        value of H_0

    w0: float
        zeroth-order parameters for dark energy equation of state

    wa: float
        first-order parameters for dark energy equation of state

    Returns
    -------
    cosmo: Colossus cosmology object
        cosmology for Colossus
    """
    cosmo_params_colossus = OrderedDict()

    if flat is not None:
        cosmo_params_colossus["flat"] = flat
    elif "flat" not in cosmo_params:
        cosmo_params["flat"] = DEFAULT_COSMOLOGY_COLOSSUS["flat"]
    else:
        cosmo_params_colossus["flat"] = cosmo_params["flat"]

    if Om0 is not None:
        cosmo_params_colossus["Om0"] = Om0
    elif "Om0" not in cosmo_params:
        cosmo_params_colossus["Om0"] = DEFAULT_COSMOLOGY_COLOSSUS["Om0"]
    else:
        cosmo_params_colossus["Om0"] = cosmo_params["Om0"]

    if sigma8 is not None:
        cosmo_params_colossus["sigma8"] = sigma8
    elif "sigma8" not in cosmo_params:
        cosmo_params_colossus["sigma8"] = DEFAULT_COSMOLOGY_COLOSSUS["sigma8"]
    else:
        cosmo_params_colossus["sigma8"] = cosmo_params["sigma8"]

    if ns is not None:
        cosmo_params_colossus["ns"] = ns
    elif "ns" not in cosmo_params:
        cosmo_params_colossus["ns"] = DEFAULT_COSMOLOGY_COLOSSUS["ns"]
    else:
        cosmo_params_colossus["ns"] = cosmo_params["ns"]

    if Ob0 is not None:
        cosmo_params_colossus["Ob0"] = Ob0
    elif "Ob0" not in cosmo_params:
        cosmo_params_colossus["Ob0"] = DEFAULT_COSMOLOGY_COLOSSUS["Ob0"]
    else:
        cosmo_params_colossus["Ob0"] = cosmo_params["Ob0"]

    if H0 is not None:
        cosmo_params_colossus["H0"] = H0
    elif "H0" not in cosmo_params:
        cosmo_params_colossus["H0"] = DEFAULT_COSMOLOGY_COLOSSUS["H0"]
    else:
        cosmo_params_colossus["H0"] = cosmo_params["H0"]

    if w0 is not None:
        cosmo_params_colossus["w0"] = w0
    elif "w0" not in cosmo_params:
        cosmo_params_colossus["w0"] = DEFAULT_COSMOLOGY_COLOSSUS["w0"]
    else:
        cosmo_params_colossus["w0"] = cosmo_params["w0"]

    if wa is not None:
        cosmo_params_colossus["wa"] = wa
    elif "wa" not in cosmo_params:
        cosmo_params_colossus["wa"] = DEFAULT_COSMOLOGY_COLOSSUS["wa"]
    else:
        cosmo_params_colossus["H0"] = cosmo_params["H0"]

    cosmo = cosmology.setCosmology(cosmo_name, **cosmo_params_colossus)

    return cosmo
