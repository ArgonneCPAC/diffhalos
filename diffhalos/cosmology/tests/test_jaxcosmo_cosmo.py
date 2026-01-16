""""""

import numpy as np

from ..jax_cosmo import Planck15, Planck18, JAXCOSMO_COSMO_PARAMS_LIST


def test_jaxcosmo_params_Planck15():
    for _param in JAXCOSMO_COSMO_PARAMS_LIST:
        assert hasattr(Planck15, _param)
        if _param != "gamma":
            assert np.isfinite(getattr(Planck15, _param))


def test_jaxcosmo_params_Planck18():
    for _param in JAXCOSMO_COSMO_PARAMS_LIST:
        assert hasattr(Planck18, _param)
        if _param != "gamma":
            assert np.isfinite(getattr(Planck18, _param))
