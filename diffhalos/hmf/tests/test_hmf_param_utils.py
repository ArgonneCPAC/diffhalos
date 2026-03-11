""" """

import numpy as np

from .. import hmf_param_utils


def test_define_diffsky_hmf_params_namedtuple_from_array():

    hmf_ntup = hmf_param_utils.define_diffsky_hmf_params_namedtuple_from_array(
        hmf_param_utils.HMF_PARAMS_ARRAY_DEFAULT
    )

    assert isinstance(hmf_ntup, tuple)
    for _field in hmf_ntup._fields:
        assert np.all(np.isfinite(getattr(hmf_ntup, _field)))


def test_define_diffsky_hmf_params_array_from_namedtuple():

    params_ntup = hmf_param_utils.DEFAULT_HMF_PARAMS

    params_array = hmf_param_utils.define_diffsky_hmf_params_array_from_namedtuple(
        params_ntup
    )

    assert params_array.shape == (hmf_param_utils.N_DIFFSKY_HMF_PARAMS,)
    assert np.all(np.isfinite(params_array))
    assert not np.all(params_array == 0)
