""" """

import numpy as np

from .. import sigmoid_utils as sut


def test_sigmoid_inversion():
    xarr = np.linspace(-10, 10, 500)

    x0, k, ylo, yhi = 0, 0.1, -5, 5
    y = sut._sigmoid(xarr, x0, k, ylo, yhi)
    x2 = sut._inverse_sigmoid(y, x0, k, ylo, yhi)
    assert np.allclose(xarr, x2, rtol=1e-4)
