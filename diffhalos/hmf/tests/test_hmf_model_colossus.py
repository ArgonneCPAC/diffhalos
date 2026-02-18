""""""

import numpy as np

from colossus.cosmology import cosmology

from ..hmf_model_colossus import colossus_diff_hmf, colossus_cuml_hmf


def test_colossus_hmf_call():
    cosmology.setCosmology("planck18")
    cosmo = cosmology.getCurrent()

    logMhalo = np.linspace(8.5, 15.5, 100)
    z = np.array([0.0, 1.5, 2.5, 3.5, 5.0])

    hmf_cut = 1e-8

    for zi in z:
        logmfunc, logm = colossus_diff_hmf(
            logMhalo,
            zi,
            cosmo,
            mdef="200m",
            model="tinker08",
            hmf_cut=hmf_cut,
        )

        assert len(logmfunc) == len(logm)
        assert np.all(np.isfinite(logmfunc))
        assert np.all(np.isfinite(logm))
        assert np.all(logmfunc > np.log10(hmf_cut))


def test_colossus_hmf_cuml():
    cosmology.setCosmology("planck18")
    cosmo = cosmology.getCurrent()

    logMhalo = np.linspace(8.5, 15.5, 100)
    z = np.array([0.0, 1.5, 2.5, 3.5, 5.0])

    hmf_cut = 1e-8

    for zi in z:
        logcumlmfunc, logm = colossus_cuml_hmf(
            logMhalo,
            zi,
            cosmo,
            mdef="200m",
            model="tinker08",
            hmf_cut=hmf_cut,
        )

        assert len(logcumlmfunc) == len(logm)
        assert np.all(np.isfinite(logcumlmfunc))
        assert np.all(np.isfinite(logm))
