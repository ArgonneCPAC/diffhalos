"""Integration utilities"""

from jax import jit as jjit
from jax.lax import scan

__all__ = (
    "cumtrapz",
    "trapz",
)


@jjit
def _cumtrapz_scan_func(carryover, el):
    b, fb = el
    a, fa, cumtrapz = carryover
    cumtrapz = cumtrapz + (b - a) * (fb + fa) / 2.0
    carryover = b, fb, cumtrapz
    accumulated = cumtrapz
    return carryover, accumulated


@jjit
def cumtrapz(xarr, yarr):
    """Cumulative trapezoidal integral

    Parameters
    ----------
    xarr : ndarray, shape (n, )

    yarr : ndarray, shape (n, )

    Returns
    -------
    result : ndarray, shape (n, )

    """
    res_init = xarr[0], yarr[0], 0.0
    scan_data = xarr, yarr
    cumtrapz = scan(_cumtrapz_scan_func, res_init, scan_data)[1]
    return cumtrapz


@jjit
def trapz(xarr, yarr):
    """Trapezoidal integral

    Parameters
    ----------
    xarr : ndarray, shape (n, )

    yarr : ndarray, shape (n, )

    Returns
    -------
    result : float

    """
    res_init = xarr[0], yarr[0], 0.0
    scan_data = xarr, yarr
    cumtrapz = scan(_cumtrapz_scan_func, res_init, scan_data)[1]
    return cumtrapz[-1]
