"""
Useful default values
"""

import numpy as np
from jax import numpy as jnp

from .cosmology import flat_wcdm, DEFAULT_COSMOLOGY_DSPS

# age of the Universe at z=0, in Gyr
T0_GYR = 13.8
LOGT0_GYR = np.log10(T0_GYR)

# full sky area
FULL_SKY_AREA = (4 * jnp.pi) * (180 / jnp.pi) ** 2

# age of the Universe at z=0
TODAY = flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY_DSPS)

# default MLP model
DEFAULT_MLP_MODEL = "mlp_model_v0"

# default diffmahnet central model
DEFAULT_DIFFMAHNET_CEN_MODEL = "cenflow_v2_0_64bit.eqx"

# default diffmahnet satellite model
DEFAULT_DIFFMAHNET_SAT_MODEL = "satflow_v2_0_64bit.eqx"
