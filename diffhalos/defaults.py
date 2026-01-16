"""
Useful default values
"""

import numpy as np
from jax import numpy as jnp
from dsps.cosmology import flat_wcdm, DEFAULT_COSMOLOGY

# age of the Universe at z=0, in Gyr
T0_GYR = 13.8
LOGT0_GYR = np.log10(T0_GYR)

# full sky area
FULL_SKY_AREA = (4 * jnp.pi) * (180 / jnp.pi) ** 2

# age of the Universe at z=0
TODAY = flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY)

# default overdensity threshold
DELTA_C = 200.0
