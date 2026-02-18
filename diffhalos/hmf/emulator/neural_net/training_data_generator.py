"""Generates training data"""

import numpy as np
from copy import deepcopy
import os

from diffsky.mass_functions.hmf_model import DEFAULT_HMF_PARAMS as P_INIT

from ..hmf_models.colossus_hmf import colossus_diff_hmf, colossus_cuml_hmf
from ..fitting_tools import hmf_fitter
from ..param_utils.defaults import DEFAULT_COSMOLOGY, DEFAULT_COSMO_PRIORS
from ..param_utils.cosmo_params import (
    define_colossus_cosmology,
    sample_cosmo_params,
)


__all__ = (
    "get_file_naming_conventions",
    "get_hmf_training_data",
    "get_best_fit_hmf_params",
    "load_cosmo_to_hmf_param_training_data",
    "load_hmf_fitter_loss_data",
)

DEFAULT_FILE_NAME_CONVENTIONS = {
    "cosmo_param_names": "cosmo_param_names.txt",
    "cosmo_param_values": "cosmo_params.npy",
    "redshift": "redshift.npy",
    "logmhalo": "logmhalo.npy",
    "loghmf": "loghmf.npy",
    "ytp_params": "ytp_params.npy",
    "x0_params": "x0_params.npy",
    "lo_params": "lo_params.npy",
    "hi_params": "hi_params.npy",
    "loss_hist": "loss_hist.npy",
    "fit_terminates": "fit_terminates.npy",
    "info": "info.txt",
}

MDEF = "200m"
HMF_MODEL = "tinker08"
HMF_CUT = 1e-8


def get_file_naming_conventions():
    """
    Convenience function to get the naming
    conventions for files to be saved and loaded
    """
    return DEFAULT_FILE_NAME_CONVENTIONS


def get_hmf_training_data(
    logmhalo,
    z,
    cuml=False,
    cosmo_params=None,
    cosmo_param_names=None,
    base_cosmo_params=DEFAULT_COSMOLOGY,
    cosmo_priors=DEFAULT_COSMO_PRIORS,
    cosmo_params_sample_method="LatinHypercube",
    seed=None,
    num_samples=None,
    mdef=MDEF,
    model=HMF_MODEL,
    hmf_cut=HMF_CUT,
    savedir=None,
    save_base_name=None,
    return_outputs=False,
    file_name_conventions=DEFAULT_FILE_NAME_CONVENTIONS,
):
    """
    Generate data files for halo mass function fits
    for a range of cosmologies and redshifts
    and save to file. These data can be used as input to
    the neural net that is trained to predict halo mass function
    parameters conditioned on the cosmological parameters or
    on the halo mass functions themselves.

    Parameters
    ----------
    logmhalo: ndarray of shape (n_halo,)
        base-10 log of halo masses, in Msun

    z: ndarray of shape (num_redshift,)
        redshift values

    cuml: bool
        if True, the cumulative HMF will be used,
        if False, the differential HMF will be used

    cosmo_params: ndarray of shape (num_samples, num_cosmo_params)
        cosmological parameter sets for ``num_samples`` points,
        for all of the ``num_cosmo_params`` cosmological
        parameters to be sampled;
        this must be provided if ``cosmo_priors`` and
        ``num_samples`` are not given

    cosmo_param_names: tuple(str)
        tuple with cosmological parameter names,
        in the order they appear in ``cosmo``;
        this must be provided if ``cosmo_priors`` is not given

    base_cosmo_params: dictionary
        base parameter dictionary;
        parameters in ``cosmo_param_names`` will be replaced
        in this dictionary to define the various sampled cosmologies
        and to define the Colossus cosmologies

    cosmo_priors: dictionary
        cosmological parameter values;
        this must be provided if ``cosmo_params`` is not given

    cosmo_params_sample_method: str
        mathod for sampling the cosmological parameter space

    seed: int
        randomly generated key

    num_samples: int
        number of cosmology samples to generate;
        this must be provided if ``cosmo_params`` is not given

    mdef: str
        mass definition

    model: str
        model for halo mass function

    hmf_cut: float
        low limit to HMF below which the halos are discarded

    savedir: str
        path to directory where the data will be stored

    save_base_name: str
        base name for output files

    return_outputs: bool
        if True, the saved data will be returned

    file_name_conventions: dictionary
        naming conventions for files

    Returns
    -------
    saves files to the required destonation and/or
    simply returns the results

    data being retunred:
    loss_data: list
        each element is a list of
        [redshift,
         base-10 log of halo mass,
         base-10 log of halo mass function]

    files being generated:
    `savedir/save_base_name`_cosmo_param_names.txt
        names of the varied cosmological parameters
        as a tuple(str)

    `savedir/save_base_name`_cosmo_param_values.npy
        values of the sampled cosmological parameters
        as a ndarray of shape (num_samples, num_cosmo_params)

    `savedir/save_base_name`_redshift.npy
        values of redshifts at which data is generated for each cosmology
        as a ndarray of shape (num_redhisft,)

    `savedir/save_base_name`_logmhalo.npy
        values of base-10 log of the halo masses
        as a ndarray of shape (num_samples, num_redshift, num_halo)

    `savedir/save_base_name`_loghmf.npy
        values of base-10 log of the halo mass function
        as a ndarray of shape (num_samples, num_redshift, num_halo)
    """
    if cosmo_params is not None and cosmo_param_names is not None:
        num_samples = cosmo_params.shape[0]
    elif num_samples is not None and cosmo_priors is not None:
        cosmo_param_names = list(cosmo_priors.keys())
        cosmo_params = sample_cosmo_params(
            cosmo_priors=cosmo_priors,
            seed=seed,
            num_samples=num_samples,
            method=cosmo_params_sample_method,
        )
    else:
        print("!ERROR! Either ``cosmo_params`` and ``cosmo_param_names``")
        print("or ``num_samples`` and ``cosmo_priors``")
        errmsg = "must be provided"
        raise Exception(errmsg)

    hmf_loss_data = _get_hmf_training_data(
        logmhalo,
        z,
        cuml=cuml,
        cosmo_params=cosmo_params,
        cosmo_param_names=cosmo_param_names,
        base_cosmo_params=base_cosmo_params,
        mdef=mdef,
        model=model,
        hmf_cut=hmf_cut,
    )

    # save outputs to files
    if savedir is not None:
        if savedir[-1] != "/":
            savedir += "/"
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        # save cosmological parameter names
        np.savetxt(
            savedir + save_base_name + "_" + file_name_conventions["cosmo_param_names"],
            cosmo_param_names,
            fmt="%s",
            header="cosmological parameter names",
        )

        # save cosmological parameter values
        np.save(
            savedir
            + save_base_name
            + "_"
            + file_name_conventions["cosmo_param_values"],
            cosmo_params,
        )

        # save redshifts
        np.save(
            savedir + save_base_name + "_" + file_name_conventions["redshift"],
            z,
        )

        # save base-10 log of halo mass and halo mass function
        # per cosmology for all redshifts together
        num_halo = len(logmhalo)
        num_redshift = len(z)

        logmhalo_data = np.zeros((num_samples, num_redshift, num_halo))
        loghmf_data = np.zeros((num_samples, num_redshift, num_halo))
        for ci in range(num_samples):
            # ---save halo masses and halo mass functions
            _loss_data = hmf_loss_data[ci]
            for zi in range(len(z)):
                _, logmp_bins, loghmf_target = _loss_data[zi]
                logmhalo_data[ci, zi, :] = logmp_bins
                loghmf_data[ci, zi, :] = loghmf_target

            np.save(
                savedir + save_base_name + "_" + file_name_conventions["logmhalo"],
                logmhalo_data,
            )
            np.save(
                savedir + save_base_name + "_" + file_name_conventions["loghmf"],
                loghmf_data,
            )

        # save additional information
        _info_n_cosmo_params = np.array(["n_cosmo_params:", cosmo_params.shape[1]])
        _info_n_cosmo_samples = np.array(["n_cosmo_params:", cosmo_params.shape[0]])
        _info_z_min = np.array(["z_min:", z[0]])
        _info_z_max = np.array(["z_max:", z[-1]])
        _info_n_z = np.array(["n_z:", num_redshift])
        _info_mh_min = np.array(["mhalo_min:", logmhalo[0]])
        _info_mh_max = np.array(["mhalo_max:", logmhalo[-1]])
        _info_n_halo = np.array(["n_halo:", num_halo])
        additional_info = np.column_stack(
            (
                _info_n_cosmo_params,
                _info_n_cosmo_samples,
                _info_z_min,
                _info_z_max,
                _info_n_z,
                _info_mh_min,
                _info_mh_max,
                _info_n_halo,
            )
        ).T
        np.savetxt(
            savedir + save_base_name + "_" + file_name_conventions["info"],
            additional_info,
            fmt="%s",
        )

    if return_outputs or savedir is None:
        return hmf_loss_data

    return


def _get_hmf_training_data(
    logMhalo,
    z,
    cuml=False,
    cosmo_params=None,
    cosmo_param_names=None,
    base_cosmo_params=DEFAULT_COSMOLOGY,
    mdef=MDEF,
    model=HMF_MODEL,
    hmf_cut=HMF_CUT,
):
    """
    Convenience function to generate training data
    for differential halo mass function,
    for multiple cosmologies and at multiple redshifts

    Parameters
    ----------
    logMhalo: ndarray of shape (n_halo,)
        base-10 log of halo masses, in Msun

    z: ndarray of shape (n_z,)
        redshift values

    cuml: bool
        if True, the cumulative HMF will be used,
        if False, the differential HMF will be used

    cosmo_params: ndarray of shape (num_samples, num_cosmo_params)
        cosmological parameter sets for ``num_samples`` points,
        for all of the ``num_cosmo_params`` cosmological
        parameters to be sampled

    cosmo_param_names: tuple(str)
        tuple with cosmological parameter names,
        in the order they appear in ``cosmo``

    base_cosmo_params: dictionary
        base parameter dictionary;
        parameters in ``cosmo_param_names`` will be replaced
        in this dictionary to define the various sampled cosmologies
        and to define the Colossus cosmologies

    mdef: str
        mass definition

    model: str
        model for halo mass function

    hmf_cut: float
        low limit to HMF below which the halos are discarded

    Returns
    -------
    loss_data: list
        each element is a list of
        [redshift,
         base-10 log of halo mass,
         base-10 log of halo mass function]
    """
    if cosmo_param_names is None or cosmo_params is None:
        single_cosmo = True
    else:
        single_cosmo = False

    # number of sampled cosmologies
    if not single_cosmo:
        num_samples = cosmo_params.shape[0]
    else:
        num_samples = 1

    # get loss data
    loss_data = []
    for ci in range(num_samples):

        # define current Colossus cosmology
        cosmo_params_cur = deepcopy(base_cosmo_params)
        if not single_cosmo:
            for pi, _param in enumerate(cosmo_param_names):
                cosmo_params_cur[_param] = cosmo_params[ci, pi]
        cosmo = define_colossus_cosmology(cosmo_params=cosmo_params_cur)

        # for current cosmology collect all redshifts
        loss_data_cosmo_cur = []
        for zi in z:
            if cuml:
                logmfunc_i, logm_i = colossus_cuml_hmf(
                    logMhalo,
                    zi,
                    cosmo,
                    mdef=mdef,
                    model=model,
                    hmf_cut=hmf_cut,
                )
            else:
                logmfunc_i, logm_i = colossus_diff_hmf(
                    logMhalo,
                    zi,
                    cosmo,
                    mdef=mdef,
                    model=model,
                    hmf_cut=hmf_cut,
                )
            loss_data_cosmo_cur.append([zi, logm_i, logmfunc_i])

        # collect loss for all cosmologies and all redshifts
        loss_data.append(loss_data_cosmo_cur)

    if num_samples == 1:
        loss_data = loss_data[0]

    return loss_data


def get_best_fit_hmf_params(
    loss_data,
    num_steps=1000,
    step_size=0.01,
    n_warmup=1,
    p_init=P_INIT,
    cuml=False,
    savedir=None,
    save_base_name=None,
    return_outputs=False,
    file_name_conventions=DEFAULT_FILE_NAME_CONVENTIONS,
):
    """
    Generate data files for halo mass function fits
    for a range of cosmologies and redshifts
    and save to file. This function runs fits to
    the input loss data and saves the results to file.

    Parameters
    ----------
    loss_data: list
        each element is a list of
        [redshift,
         base-10 log of halo mass,
         base-10 log of halo mass function]

    num_steps: int
        number of steps for the fitter

    step_size: float
        step size for the fitter

    n_warmup: int
        number of warmup steps for the fitter

    p_init: namedtuple
        initial guess of HMF parameters

    cuml: bool
        if True, the cumulative HMF will be used,
        if False, the differential HMF will be used

    savedir: str
        path to directory where the data will be stored

    save_base_name: str
        base name for output files

    return_outputs: bool
        if True, the saved data will be returned

    file_name_conventions: dictionary
        naming conventions for files

    Returns
    -------
    saves files to the required destonation and/or
    simply returns the results

    data being retunred:
    res: list
        contains the results from all cosmology fits
        as tuples with the following

        p_best: namedtuple
            ``HMF_Params`` named tuple with the
            best-fit diffsky HMF parameters

        loss: float
            loss at the last step

        loss_hist: ndarray of shape (n_steps,)
            loss at each step

        params_hist: list of length ``n_steps``
            HMF parameters at each step

        fit_terminates:
            0 if NaN or inf is encountered by the fitter,
            causing termination before n_step, or
            1 for a fit that terminates with no such problems

    files being generated:
    `savedir/save_base_name`_ytp_params.npy
        values of the ``ytp_params`` field in the
        ``HMF_Params`` namedtuple holding the HMF diffsky parameters
        as a ndarray of shape (num_samples, num_ytp)

    `savedir/save_base_name`_x0_params.npy
        values of the ``x0_params`` field in the
        ``HMF_Params`` namedtuple holding the HMF diffsky parameters
        as a ndarray of shape (num_samples, num_x0)

    `savedir/save_base_name`_lo_params.npy
        values of the ``lo_params`` field in the
        ``HMF_Params`` namedtuple holding the HMF diffsky parameters
        as a ndarray of shape (num_samples, num_lo)

    `savedir/save_base_name`_hi_params.npy
        values of the ``hi_params`` field in the
        ``HMF_Params`` namedtuple holding the HMF diffsky parameters
        as a ndarray of shape (num_samples, num_hi)

    `savedir/save_base_name`_loss_hist.npy
        values of the loss at each step of the gradient decent
        as a ndarray of shape (num_samples, num_steps)

    `savedir/save_base_name`_fit_terminates.npy
        values of the exit status of each fit
        as a ndarray of shape (num_samples, )
    """
    res = hmf_fitter.fit_hmf_multi_cosmo(
        loss_data,
        n_steps=num_steps,
        step_size=step_size,
        n_warmup=n_warmup,
        p_init=p_init,
        cuml=cuml,
    )

    # save outputs to files
    if savedir is not None:
        if savedir[-1] != "/":
            savedir += "/"
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        # ---save best-fit parameters and loss history
        num_samples = len(loss_data)

        ytp_params_all = np.zeros((num_samples, 5))
        x0_params_all = np.zeros((num_samples, 5))
        lo_params_all = np.zeros((num_samples, 4))
        hi_params_all = np.zeros((num_samples, 5))
        loss_hist_all = np.zeros((num_samples, num_steps))
        fit_terminates_all = np.zeros((num_samples, num_steps))

        for ci in range(num_samples):
            p_best, loss, loss_hist, params_hist, fit_terminates = res[ci]

            ytp_params_all[ci, :] = np.asarray(p_best._asdict()["ytp_params"])
            x0_params_all[ci, :] = np.asarray(p_best._asdict()["x0_params"])
            lo_params_all[ci, :] = np.asarray(p_best._asdict()["lo_params"])
            hi_params_all[ci, :] = np.asarray(p_best._asdict()["hi_params"])
            loss_hist_all[ci, :] = loss_hist
            fit_terminates_all[ci, :] = fit_terminates

        np.save(
            savedir + save_base_name + "_" + file_name_conventions["ytp_params"],
            ytp_params_all,
        )
        np.save(
            savedir + save_base_name + "_" + file_name_conventions["x0_params"],
            x0_params_all,
        )
        np.save(
            savedir + save_base_name + "_" + file_name_conventions["lo_params"],
            lo_params_all,
        )
        np.save(
            savedir + save_base_name + "_" + file_name_conventions["hi_params"],
            hi_params_all,
        )
        np.save(
            savedir + save_base_name + "_" + file_name_conventions["loss_hist"],
            loss_hist_all,
        )
        np.save(
            savedir + save_base_name + "_" + file_name_conventions["fit_terminates"],
            fit_terminates_all,
        )

    if return_outputs or savedir is None:
        return res

    return


def load_cosmo_to_hmf_param_training_data(
    savedir_input=None,
    savedir_target=None,
    save_base_name_input=None,
    save_base_name_target=None,
    file_name_conventions=DEFAULT_FILE_NAME_CONVENTIONS,
):
    """
    Convenience function to load data for training
    a neural network to predict halo mass funtion
    parameters from inppt cosmological parameters. This
    function loads the input cosmology and output best-fit
    halo mass function parameters,
    conditioned on the redshift values chosen to fit.

    Parameters
    ----------
    savedir_input: str
        path to directory where the input data is stored

    savedir_target: str
        path to directory where the target data is stored

    save_base_name_input: str
        base name of input data files to load

    save_base_name_target: str
        base name of target data files to load

    file_name_conventions: dictionary
        naming conventions for files

    Returns
    -------
    cosmo_param_names: list(str)
        names of the varied cosmological parameters

    cosmo_param_values: ndarray of shape (num_samples, num_cosmo_params)
        values of the sampled cosmological parameters

    z: ndarray of shape (num_redshift, )
        values of redshifts at which data is generated for each cosmology

    ytp_params: ndarray of shape (num_samples, num_ytp)
        values of the ``ytp_params`` field in the
        ``HMF_Params`` namedtuple holding the HMF diffsky parameters

    x0_params: ndarray of shape (num_samples, num_x0)
        values of the ``x0_params`` field in the
        ``HMF_Params`` namedtuple holding the HMF diffsky parameters

    lo_params: ndarray of shape (num_samples, num_lo)
        values of the ``lo_params`` field in the
        ``HMF_Params`` namedtuple holding the HMF diffsky parameters

    hi_params: ndarray of shape (num_samples, num_hi)
        values of the ``hi_params`` field in the
        ``HMF_Params`` namedtuple holding the HMF diffsky parameters

    hmf_params_all: ndarray of shape (num_samples, num_hmf_all)
        values of all of the HMF paramters concatenated

    logmhalo: ndarray of shape (num_samples, num_redshift, num_halo)
        values of base-10 log of the halo masses

    loghmf: ndarray of shape (num_samples, num_redshift, num_halo)
        values of base-10 log of the halo mass function
    """
    # load files
    if savedir_input[-1] != "/":
        savedir_input += "/"
    if savedir_target[-1] != "/":
        savedir_target += "/"

    # ---load cosmological parameters
    cosmo_param_names = np.loadtxt(
        savedir_input
        + save_base_name_input
        + "_"
        + file_name_conventions["cosmo_param_names"],
        dtype=str,
    )

    cosmo_param_values = np.load(
        savedir_input
        + save_base_name_input
        + "_"
        + file_name_conventions["cosmo_param_values"],
    )

    # ---load redshift values
    z = np.load(
        savedir_input + save_base_name_input + "_" + file_name_conventions["redshift"]
    )

    # --load halo mass function parameters
    ytp_params = np.load(
        savedir_target
        + save_base_name_target
        + "_"
        + file_name_conventions["ytp_params"],
    )
    x0_params = np.load(
        savedir_target
        + save_base_name_target
        + "_"
        + file_name_conventions["x0_params"],
    )
    lo_params = np.load(
        savedir_target
        + save_base_name_target
        + "_"
        + file_name_conventions["lo_params"],
    )
    hi_params = np.load(
        savedir_target
        + save_base_name_target
        + "_"
        + file_name_conventions["hi_params"],
    )

    # combine all HMF parameters into a signle array
    num_samples = cosmo_param_values.shape[0]
    num_ypt_params = ytp_params.shape[1]
    num_x0_params = x0_params.shape[1]
    num_lo_params = lo_params.shape[1]
    num_hi_params = hi_params.shape[1]

    num_hmf_all = num_ypt_params + num_x0_params + num_lo_params + num_hi_params
    hmf_params_all = np.zeros((num_samples, num_hmf_all))
    for i in range(num_samples):
        hmf_params_all[i, :] = (
            list(ytp_params[i, :])
            + list(x0_params[i, :])
            + list(lo_params[i, :])
            + list(hi_params[i, :])
        )

    # ---load halo mass and halo mass function
    logmhalo = np.load(
        savedir_input + save_base_name_input + "_" + file_name_conventions["logmhalo"]
    )
    loghmf = np.load(
        savedir_input + save_base_name_input + "_" + file_name_conventions["loghmf"]
    )

    return (
        cosmo_param_names,
        cosmo_param_values,
        z,
        ytp_params,
        x0_params,
        lo_params,
        hi_params,
        hmf_params_all,
        logmhalo,
        loghmf,
    )


def load_hmf_fitter_loss_data(
    savedir_input=None,
    save_base_name_input=None,
    file_name_conventions=DEFAULT_FILE_NAME_CONVENTIONS,
):
    """
    Convenience function to load data for running
    the halo mass function fitter in the right format

    Parameters
    ----------
    savedir_input: str
        path to directory where the input data is stored

    save_base_name_input: str
        base name of input data files to load

    file_name_conventions: dictionary
        naming conventions for files

    Returns
    -------
    loss_data: list
        each element is a list of
        [redshift,
         base-10 log of halo mass,
         base-10 log of halo mass function]
    """
    # load files
    if savedir_input[-1] != "/":
        savedir_input += "/"

    # load redshift values and cosmologies
    z = np.load(
        savedir_input + save_base_name_input + "_" + file_name_conventions["redshift"]
    )
    cosmo_param_values = np.load(
        savedir_input
        + save_base_name_input
        + "_"
        + file_name_conventions["cosmo_param_values"],
    )
    num_samples = cosmo_param_values.shape[0]

    # load halo mass and halo mass function
    logmhalo = np.load(
        savedir_input + save_base_name_input + "_" + file_name_conventions["logmhalo"]
    )
    loghmf = np.load(
        savedir_input + save_base_name_input + "_" + file_name_conventions["loghmf"]
    )

    loss_data = []
    for ic in range(num_samples):
        for iz, zi in enumerate(z):
            _loss = [zi, logmhalo[iz, iz, :], loghmf[ic, iz, :]]
        loss_data.append(_loss)

    return loss_data
