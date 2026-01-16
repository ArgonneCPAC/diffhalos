""""""

import numpy as np

from ..dsps_cosmo import DEFAULT_COSMOLOGY_DSPS, DSPS_COSMO_PARAMS_LIST


def test_dsps_cosmo_params_namedtuple():
    for _field in DEFAULT_COSMOLOGY_DSPS._fields:
        assert _field in DSPS_COSMO_PARAMS_LIST
        assert np.isfinite(DEFAULT_COSMOLOGY_DSPS._asdict()[_field])
