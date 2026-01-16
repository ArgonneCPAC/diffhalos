"""
Convenience functions for DSPS cosmological parameters
"""

from dsps.cosmology import PLANCK15, WMAP5, FSPS_COSMO  # noqa
from dsps.cosmology import PLANCK15 as DEFAULT_COSMOLOGY_DSPS  # noqa

DSPS_COSMO_PARAMS_LIST = (
    "Om0",
    "w0",
    "wa",
    "h",
)
