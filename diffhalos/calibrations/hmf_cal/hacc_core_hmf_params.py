"""
Differential HMF fit to the core mass function including cens and all cores.
Analogous to the unevolved subhalo function.

These best-fitting parameters were derived with the
``refit_central_hmf_to_discovery.ipynb`` notebook
"""

from .smdpl_hmf import HMF_PARAMS as SMDPL_HMF_PARAMS

ytp_params = SMDPL_HMF_PARAMS.ytp_params._replace(
    ytp_ytp=-5.640,
    ytp_x0=0.683,
    ytp_k=0.812,
    ytp_ylo=-0.315,
    ytp_yhi=-1.449,
)

x0_params = SMDPL_HMF_PARAMS.x0_params._replace(
    x0_ytp=12.953,
    x0_x0=2.137,
    x0_k=3.267,
    x0_ylo=-1.093,
    x0_yhi=-0.300,
)

lo_params = SMDPL_HMF_PARAMS.lo_params._replace(
    lo_x0=3.576,
    lo_k=0.729,
    lo_ylo=-0.816,
    lo_yhi=-2.842,
)

hi_params = SMDPL_HMF_PARAMS.hi_params._replace(
    hi_ytp=-4.770,
    hi_x0=3.880,
    hi_k=1.450,
    hi_ylo=-0.127,
    hi_yhi=-1.033,
)

HMF_PARAMS = SMDPL_HMF_PARAMS._make(
    (ytp_params, x0_params, lo_params, hi_params),
)
