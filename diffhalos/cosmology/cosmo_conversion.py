"""
Convenience functions to convert between
jax-cosmo and dsps cosmologies
"""

from functools import partial

from jax import jit as jjit

from .dsps_cosmo import DEFAULT_COSMOLOGY_DSPS
from .jax_cosmo import Cosmology, DEFAULT_COSMOLOGY_JAXCOSMO

__all__ = (
    "jaxcosmo_to_dsps_cosmology",
    "dsps_to_jaxcosmo_cosmology",
)


@jjit
def jaxcosmo_to_dsps_cosmology(jax_cosmo):
    """
    Convert a jax-cosmo type cosmology to
    a dsps type of cosmology, with the same parameters

    Parameters
    ----------
    jax_cosmo: jax-cosmo parameters object
        cosmological paramteters parameters

    Returns
    -------
    dsps_cosmo: namedtuple
        dsps.cosmology.flat_wcdm cosmology
    """
    dsps_cosmo = DEFAULT_COSMOLOGY_DSPS._replace(
        Om0=jax_cosmo.Omega_b + jax_cosmo.Omega_c,
        w0=jax_cosmo.w0,
        wa=jax_cosmo.wa,
        h=jax_cosmo.h,
    )

    return dsps_cosmo


@jjit
def dsps_to_jaxcosmo_cosmology(
    dsps_cosmo,
    Omega_b=DEFAULT_COSMOLOGY_JAXCOSMO.Omega_b,
    Omega_k=DEFAULT_COSMOLOGY_JAXCOSMO.Omega_k,
    n_s=DEFAULT_COSMOLOGY_JAXCOSMO.n_s,
    sigma8=DEFAULT_COSMOLOGY_JAXCOSMO.sigma8,
    gamma=DEFAULT_COSMOLOGY_JAXCOSMO.gamma,
):
    """
    Convert a jax-cosmo type cosmology to
    a dsps type of cosmology, with the same parameters

    Parameters
    ----------
    dsps_cosmo: namedtuple
        dsps.cosmology.flat_wcdm cosmology

    Omega_b: float
        baryonic matter density fraction

    Omega_k: float
        curvature density fraction

    n_s: float
        primordial scalar perturbation spectral index

    sigma8: float
        variance of matter density perturbations at an 8 Mpc/h scale

    gamma: float
        index of the growth rate

    Returns
    -------
    jaxcosmo: jax-cosmo parameters object
        cosmological paramteters parameters
    """
    jax_cosmo = partial(
        Cosmology,
        Omega_c=dsps_cosmo.Om0 - Omega_b,
        Omega_b=Omega_b,
        Omega_k=Omega_k,
        h=dsps_cosmo.h,
        n_s=n_s,
        sigma8=sigma8,
        w0=dsps_cosmo.w0,
        wa=dsps_cosmo.wa,
        gamma=gamma,
    )()

    return jax_cosmo
