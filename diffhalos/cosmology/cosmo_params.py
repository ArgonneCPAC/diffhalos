"""Cosmological parameter utils"""

import numpy as np
from scipy.stats import qmc

from colossus.cosmology import cosmology

from .defaults import DEFAULT_COSMOLOGY, DEFAULT_COSMO_PRIORS

__all__ = ("define_colossus_cosmology", "sample_cosmo_params")


AV_SAMPLING_METHODS = ("LatinHypercube",)


def define_colossus_cosmology(
    cosmo_name="cosmo",
    cosmo_params=DEFAULT_COSMOLOGY,
    flat=None,
    Om0=None,
    sigma8=None,
    ns=None,
    Ob0=None,
    H0=None,
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

    Returns
    -------
    cosmo: Colossus cosmology object
        cosmology for Colossus
    """
    if flat is not None:
        cosmo_params["flat"] = flat
    if Om0 is not None:
        cosmo_params["Om0"] = Om0
    if sigma8 is not None:
        cosmo_params["sigma8"] = sigma8
    if ns is not None:
        cosmo_params["ns"] = ns
    if Ob0 is not None:
        cosmo_params["Ob0"] = Ob0
    if H0 is not None:
        cosmo_params["H0"] = H0

    cosmo = cosmology.setCosmology(cosmo_name, **cosmo_params)

    return cosmo


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

    return cosmo_samples


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
