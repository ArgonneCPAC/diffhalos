"""Script to produce the HMF training data"""

import numpy as np
from collections import OrderedDict
import configparser
import argparse

from ..fitting_tools import training_data_generator
from ..utils import MyConfigParser


"""Read the arguments passed in command line"""
# initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--ini",
    required=False,
    type=str,
    default="settings.ini",
    help="Enter the name of the ini file to use for this run",
)
parser.add_argument(
    "--include",
    required=False,
    type=str,
    default="EMPTY",
    help="Enter the name of any additional ini files to include",
)
parse_args = parser.parse_args()

# setup parsed information that will be passed to configparser
inifilelist = []

incfile = parse_args.include
if incfile == "EMPTY":
    pass
else:
    if not incfile.endswith(".ini"):
        incfile += ".ini"
    inifilelist.append(incfile)

inifile = parse_args.ini
if not inifile.endswith(".ini"):
    inifile += ".ini"
inifilelist.append(inifile)

"""Import settings from ini files"""
print("Reading ini files")
# load configuration settings
config = configparser.ConfigParser()
config.optionxform = str
config.read(inifilelist)
myconfig = MyConfigParser(config)
print("Done.\n")

"""Setup training data run"""
print("Setting up training data generation")

# get redshift values
if config["training.data"]["z_array"] == "":
    z_min = config["training.data"].getfloat("z_min")
    z_max = config["training.data"].getfloat("z_max")
    n_z = config["training.data"].getint("n_z")
    z = np.linspace(z_min, z_max, n_z)
else:
    z = np.asarray(myconfig.getlistfloat("training.data", "z_array"))

# get halo mass values
logmhalo_min = config["training.data"].getfloat("logmhalo_min")
logmhalo_max = config["training.data"].getfloat("logmhalo_max")
n_mhalo = config["training.data"].getint("n_halo")
logmhalo = np.linspace(logmhalo_min, logmhalo_max, n_mhalo)
mhalo = 10**logmhalo

# get hmf model settings
mdef = config["hmf.model"]["mdef"]
model = config["hmf.model"]["model"]
q_out = config["hmf.model"]["q_out"]
h_units = config["hmf.model"].getboolean("h_units")

# get hmf fitter settings
num_steps = config["hmf.fitter"].getint("num_steps")
step_size = config["hmf.fitter"].getfloat("step_size")
n_warmup = config["hmf.fitter"].getint("n_warmup")
hmf_cut = config["hmf.fitter"].getfloat("hmf_cut")
cuml = config["hmf.fitter"].getboolean("cuml")

# get cosmology sampling setting
if config["training.data"]["seed"] == "":
    seed = None
else:
    seed = config["training.data"]["seed"]
num_samples = config["training.data"].getint("n_cosmo")
cosmo_params_sample_method = config["training.data"]["cosmo_sampling_method"]

# get default cosmological parameters
base_cosmo_params = OrderedDict(
    flat=config["cosmo.params.defaults"].getboolean("flat"),
    Om0=config["cosmo.params.defaults"].getfloat("Om0"),
    sigma8=config["cosmo.params.defaults"].getfloat("sigma8"),
    ns=config["cosmo.params.defaults"].getfloat("ns"),
    Ob0=config["cosmo.params.defaults"].getfloat("Ob0"),
    H0=config["cosmo.params.defaults"].getfloat("H0"),
)

# get default cosmological parameter priors
cosmo_priors = OrderedDict()
for _param, _ in config.items("cosmo.params.priors"):
    _vals = myconfig.getlistfloat("cosmo.params.priors", _param)
    cosmo_priors[_param] = _vals

# get path settings
savedir_input = config["training.paths"]["savedir_input"]
savedir_target = config["training.paths"]["savedir_target"]
save_base_name_input = config["training.paths"]["save_base_name_input"]
save_base_name_target = config["training.paths"]["save_base_name_target"]

# what to generate
get_loss_data = config["training.data"].getboolean("get_loss_data")
get_hmf_best_fit = config["training.data"].getboolean("get_hmf_best_fit")

print("Done.\n")

"""Produce the training data"""
print("Running scripts to generate files")

# run the fitter and save the best-fit HMF parameters to file
if get_loss_data:
    loss_data = training_data_generator.get_hmf_training_data(
        logmhalo,
        z,
        cuml=cuml,
        base_cosmo_params=base_cosmo_params,
        cosmo_priors=cosmo_priors,
        cosmo_params_sample_method=cosmo_params_sample_method,
        seed=seed,
        num_samples=num_samples,
        mdef=mdef,
        model=model,
        q_out=q_out,
        h_units=h_units,
        hmf_cut=hmf_cut,
        savedir=savedir_input,
        save_base_name=save_base_name_input,
        return_outputs=True,
    )
else:
    loss_data = training_data_generator.load_hmf_fitter_loss_data(
        savedir_input=savedir_input,
        save_base_name_input=save_base_name_input,
    )

if get_hmf_best_fit:
    _ = training_data_generator.get_best_fit_hmf_params(
        loss_data,
        num_steps=num_steps,
        step_size=step_size,
        n_warmup=n_warmup,
        cuml=cuml,
        savedir=savedir_target,
        save_base_name=save_base_name_target,
        return_outputs=False,
    )

print("Done.\n")

print("Input data saved in %s" % savedir_input)
print("Target data saved in %s" % savedir_target)
