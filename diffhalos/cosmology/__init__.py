""""""

# flake8: noqa

from .cosmo_params import PLANCK15 as DEFAULT_COSMOLOGY
from .cosmo_params import PLANCK15, WMAP5, FSPS_COSMO, COSMOS20, CosmoParams

from .flat_wcdm import age_at_z0

TODAY = age_at_z0(*DEFAULT_COSMOLOGY)
