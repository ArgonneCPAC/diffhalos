""""""

import numpy as np

from .. import cosmo_conversion
from ..dsps_cosmo import DSPS_COSMO_PARAMS_LIST


def test_jaxcosmo_to_dsps_cosmology():

    jax_cosmo = cosmo_conversion.DEFAULT_COSMOLOGY_JAXCOSMO
    dsps_cosmo = cosmo_conversion.jaxcosmo_to_dsps_cosmology(jax_cosmo)

    for _param in DSPS_COSMO_PARAMS_LIST:
        assert _param in dsps_cosmo._fields
        assert np.isfinite(dsps_cosmo._asdict()[_param])

        if _param == "Om0":
            jaxcosmo_param = jax_cosmo.Omega_c + jax_cosmo.Omega_b
            assert dsps_cosmo._asdict()[_param] == jaxcosmo_param
        else:
            assert dsps_cosmo._asdict()[_param] == getattr(jax_cosmo, _param)


def test_dsps_to_jaxcosmo_cosmology():

    dsps_cosmo = cosmo_conversion.DEFAULT_COSMOLOGY_DSPS
    jax_cosmo = cosmo_conversion.dsps_to_jaxcosmo_cosmology(dsps_cosmo)

    for _param in DSPS_COSMO_PARAMS_LIST:
        assert _param in dsps_cosmo._fields
        assert np.isfinite(dsps_cosmo._asdict()[_param])

        if _param == "Om0":
            jaxcosmo_param = jax_cosmo.Omega_c + jax_cosmo.Omega_b
            assert dsps_cosmo._asdict()[_param] == jaxcosmo_param
        else:
            assert dsps_cosmo._asdict()[_param] == getattr(jax_cosmo, _param)
