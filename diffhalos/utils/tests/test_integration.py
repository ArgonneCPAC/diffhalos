""" """

import numpy as np
import pytest
from jax import random as jran

from ..integration import cumtrapz, trapz

try:
    from scipy.integrate import trapezoid

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

MSG_HAS_SCIPY = "Must have scipy installed to run this test"


@pytest.mark.skipif(not HAS_SCIPY, reason=MSG_HAS_SCIPY)
def test_cumtrapz():
    ran_key = jran.PRNGKey(0)
    n_x = 100
    n_tests = 10
    for __ in range(n_tests):
        x_key, y_key, ran_key = jran.split(ran_key, 3)
        xarr = np.sort(jran.uniform(x_key, minval=0, maxval=1, shape=(n_x,)))
        yarr = jran.uniform(y_key, minval=0, maxval=1, shape=(n_x,))
        jax_result = cumtrapz(xarr, yarr)
        np_result = [trapezoid(yarr[:-i], x=xarr[:-i]) for i in range(1, n_x)][::-1]
        assert np.allclose(jax_result[:-1], np_result, rtol=1e-4)
        assert np.allclose(jax_result[-1], trapezoid(yarr, x=xarr), rtol=1e-4)


@pytest.mark.skipif(not HAS_SCIPY, reason=MSG_HAS_SCIPY)
def test_trapz():
    ran_key = jran.PRNGKey(0)
    n_x = 100
    n_tests = 10
    for __ in range(n_tests):
        x_key, y_key, ran_key = jran.split(ran_key, 3)
        xarr = np.sort(jran.uniform(x_key, minval=0, maxval=1, shape=(n_x,)))
        yarr = jran.uniform(y_key, minval=0, maxval=1, shape=(n_x,))
        jax_result = trapz(xarr, yarr)
        np_result = trapezoid(yarr, x=xarr)
        assert np.allclose(jax_result, np_result, rtol=1e-4)
