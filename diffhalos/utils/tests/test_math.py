""" """

import numpy as np

from ..math import map_intervals


def test_map_intervals():

    arr = np.linspace(1, 10, 100)

    oldMin = 1
    oldMax = 10

    newMin = 5
    newMax = 30

    new_arr = map_intervals(arr, oldMin, oldMax, newMin, newMax)

    assert arr.shape == new_arr.shape
    assert np.all(new_arr >= newMin)
    assert np.all(new_arr <= newMax)
