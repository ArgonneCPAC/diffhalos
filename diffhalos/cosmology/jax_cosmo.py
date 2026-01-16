"""
Halox Halo Mass Function parameters.
This is a convenience module to load and manipulate
Halox-type jax-cosmo parameter sets, set defaults, etc.
See also: https://halox.readthedocs.io/en/latest/api/halox.cosmology.html,
https://github.com/DifferentiableUniverseInitiative/jax_cosmo/tree/master/jax_cosmo
"""

from functools import partial  # noqa
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    from jax_cosmo.core import Cosmology  # noqa

from halox import cosmology

Planck18 = cosmology.Planck18()
DEFAULT_COSMOLOGY_JAXCOSMO = Planck18

"""
To add new cosmologies, we just set the parameters
to some default values using partial
"""
# Planck 2015 paper XII Table 4 final column (best fit)
Planck15 = partial(
    Cosmology,
    Omega_c=0.2589,
    Omega_b=0.04860,
    Omega_k=0.0,
    h=0.6774,
    n_s=0.9667,
    sigma8=0.8159,
    w0=-1.0,
    wa=0.0,
    gamma=None,
)()

JAXCOSMO_COSMO_PARAMS_LIST = (
    "Omega_c",
    "Omega_b",
    "h",
    "n_s",
    "sigma8",
    "Omega_k",
    "w0",
    "wa",
    "gamma",
)
