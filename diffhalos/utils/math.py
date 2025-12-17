"""Some common math utils"""

from jax import jit as jjit


@jjit
def map_intervals(values, oldMin, oldMax, newMin, newMax):
    """
    Helper function to map values in the interval [oldMin, oldMax]
    to the new interval [newMin, newMax]
    """
    return ((values - oldMin) / (oldMax - oldMin)) * (newMax - newMin) + newMin
