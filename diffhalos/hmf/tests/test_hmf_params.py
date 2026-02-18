""" """

import numpy as np

from .. import hmf_params


def test_define_diffsky_hmf_params_namedtuple_computes():

    hmf_ntup = hmf_params.define_diffsky_hmf_params_namedtuple()

    for _field in hmf_ntup._fields:
        assert np.all(np.isfinite(getattr(hmf_ntup, _field)))
