"""Generate HMF model predictions using Colossus"""

import numpy as np

from colossus.lss import mass_function
from scipy.interpolate import UnivariateSpline


__all__ = (
    "colossus_diff_hmf",
    "colossus_cuml_hmf",
)

MDEF = "200m"
HMF_MODEL = "tinker08"
HMF_CUT = 1e-8


def colossus_diff_hmf(
    logMhalo,
    z,
    cosmo,
    mdef=MDEF,
    model=HMF_MODEL,
    hmf_cut=HMF_CUT,
):
    """
    Differential HMF from Colossus
    at a single redshift point

    Parameters
    ----------
    logMhalo: ndarray of shape (n_halo,)
        base-10 log of halo masses, in Msun

    z: float
        redshift point

    cosmo: colossus cosmology
        a Colossus cosmology object

    mdef: str
        mass definition

    model: str
        model for halo mass function

    hmf_cut: float
        low limit to HMF below which the halos are discarded

    Returns
    -------
    logmfunc_out: ndarray of shape (n_halo_out,)
        base-10 log of halo mass function model predictions after applying the
        HMF cut, in comoving 1/Mpc^3

    logMhalo_out: ndarray of shape (n_halo_out,)
        base-10 log of halo masses after applying the
        HMF cut, in Msun
    """
    # halo masses
    Mhalo = 10**logMhalo
    nhalo = len(logMhalo)

    # halo mass function
    mfunc = _call_colossus(
        Mhalo * cosmo.h,
        z,
        cosmo,
        mdef,
        model,
    )

    # cut at low HMF limit
    _filter = np.where(mfunc > hmf_cut)[0]
    if len(_filter > 0):
        logMhalo_after_cut = logMhalo[_filter]
        logm_min = logMhalo_after_cut[0]
        logm_max = logMhalo_after_cut[-1]
        logMhalo_out = np.linspace(logm_min, logm_max, nhalo)
        mfunc_new = _call_colossus(
            10**logMhalo_out * cosmo.h,
            z,
            cosmo,
            mdef,
            model,
        )
        logmfunc_out = np.log10(mfunc_new)
    else:
        logMhalo_out = logMhalo
        logmfunc_out = np.log10(mfunc)

    return logmfunc_out, logMhalo_out


def _call_colossus(
    Mhalo,
    z,
    cosmo,
    mdef,
    model,
):
    # call colossus to compute the HMF
    mfunc = mass_function.massFunction(
        Mhalo * cosmo.h, z, mdef=mdef, model=model, q_out="dndlnM"
    )

    # convert from dn/dlnM to dn/dlogM
    mfunc *= np.log(10.0)

    # remove h units
    mfunc *= cosmo.h**3

    return mfunc


def colossus_cuml_hmf(
    logMhalo,
    z,
    cosmo,
    mdef=MDEF,
    model=HMF_MODEL,
    hmf_cut=HMF_CUT,
):
    """
    Cumulative HMF from Colossus, inferred from differential
    at a single redshift point

    Parameters
    ----------
    logMhalo: ndarray of shape (n_halo,)
        base-10 log of halo masses, in Msun

    z: ndarray of shape (n_z,)
        redshift values

    cosmo: colossus cosmology
        a Colossus cosmology object

    mdef: str
        mass definition

    model: str
        model for halo mass function

    hmf_cut: float
        low limit to HMF below which the halos are discarded

    Returns
    -------
    loghmf_cuml_func: list
        base-10 log of cumulative halo mass function
        model predictions at each redshift

    logm_cuml: list
        base-10 log of halo masses after applying the
        HMF cut, in Msun
    """
    # halo mass function
    logmfunc, logm = colossus_diff_hmf(
        logMhalo, z, cosmo, mdef=mdef, model=model, hmf_cut=hmf_cut
    )

    # integrate to obtain the cumulative halo mass function
    logm_cuml = logm[:-1]
    loghmf_cuml_func = np.zeros(len(logm_cuml))
    for i in range(len(logm_cuml)):
        loghmf_cuml_func[i] = np.log10(
            UnivariateSpline(
                logm,
                10**logmfunc,
                s=0.0,
            ).integral(logm_cuml[i], logm[-1])
        )

    return loghmf_cuml_func, logm_cuml
