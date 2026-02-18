"""Defaults and convenient definitions"""

# planck 2018 cosmology
DEFAULT_COSMOLOGY = {
    "flat": True,
    "Om0": 0.3111,
    "sigma8": 0.8102,
    "ns": 0.9665,
    "Ob0": 0.0490,
    "H0": 67.66,
}

# cosmological parameter priors
DEFAULT_COSMO_PRIORS = {
    "Om0": (0.2, 0.5),
    "sigma8": (0.6, 1.0),
    "ns": (0.8, 1.1),
}
