""" """

import numpy as np

from .. import loss_functions as lf


def test_mse_works_as_expected():

    pred = np.linspace(0, 100, 100)
    target = np.linspace(0, 100, 100)

    mse = lf.mse(pred, target)
    assert mse == 0

    pred = np.linspace(0, 100, 100)
    target = np.linspace(0, 100, 100) + 1

    mse = lf.mse(pred, target)
    assert mse == 1
