""" """

import numpy as np

from .. import hmf_param_utils


def test_define_diffsky_hmf_params_namedtuple_computes():

    hmf_ntup = hmf_param_utils.define_diffsky_hmf_params_namedtuple(
        hmf_param_utils.HMF_PARAMS_ARRAY_DEFAULT
    )

    assert isinstance(hmf_ntup, tuple)
    for _field in hmf_ntup._fields:
        assert np.all(np.isfinite(getattr(hmf_ntup, _field)))
