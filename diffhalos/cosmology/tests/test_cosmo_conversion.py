""" """

import numpy as np

from .. import cosmo_conversion
from ..cosmo_dsps import DSPS_COSMO_PARAMS_LIST


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


def test_alt_default_jaxcosmo_cosmology():
    jax_cosmo_def = cosmo_conversion.DEFAULT_COSMOLOGY_JAXCOSMO
    jax_cosmo_alt = cosmo_conversion.alt_default_jaxcosmo_cosmology(
        Omega_c=0.2, sigma8=0.71, h=0.65
    )

    JAXCOSMO_COSMO_PARAMS_LIST = (
        "Omega_c",
        "Omega_b",
        "h",
        "n_s",
        "sigma8",
        "Omega_k",
        "w0",
        "wa",
    )

    _params_changed = ("Omega_c", "sigma8", "h")

    for _param in JAXCOSMO_COSMO_PARAMS_LIST:
        _val_def = getattr(jax_cosmo_def, _param)
        _val_alt = getattr(jax_cosmo_alt, _param)

        assert np.isfinite(_val_def)
        assert np.isfinite(_val_alt)

        if _param in _params_changed:
            assert _val_def != _val_alt
        else:
            assert _val_def == _val_alt
