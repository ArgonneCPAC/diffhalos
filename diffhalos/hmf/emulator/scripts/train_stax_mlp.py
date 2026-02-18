"""Script to train the MLP"""

import configparser
import argparse

from ..neural_net.mlp_stax import MLP_stax
from ..neural_net import training_data_generator, loss_functions
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

# get path settings
savedir_input = config["training.paths"]["savedir_input"]
savedir_target = config["training.paths"]["savedir_target"]
save_base_name_input = config["training.paths"]["save_base_name_input"]
save_base_name_target = config["training.paths"]["save_base_name_target"]

print("Done.\n")

"""Load the training data"""
print("Loading the training data")

(
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
) = training_data_generator.load_cosmo_to_hmf_param_training_data(
    savedir_input=savedir_input,
    savedir_target=savedir_target,
    save_base_name_input=save_base_name_input,
    save_base_name_target=save_base_name_target,
)

input_data = cosmo_param_values
target_data = hmf_params_all

n_features = input_data.shape[1]
n_targets = target_data.shape[1]

print("Done.\n")

"""Setup settings for MLP training"""
print("Setting up MLP")

seed = config["training.mlp"].getint("seed")
step_size = config["training.mlp"].getfloat("step_size")
batch_size = config["training.mlp"].getint("batch_size")
n_batches = config["training.mlp"].getint("n_batches")
n_epochs = config["training.mlp"].getint("n_epochs")
loss_function = config["training.mlp"]["loss_function"]
if loss_function == "mse":
    loss_fn = loss_functions.mse_loss
else:
    errmsg = "!ERROR! Loss function %s not valid" % loss_function
    raise Exception(errmsg)

print("Done.\n")

"""Produce the training data"""
print("Running scripts to train the MLP")

# define and initialize the MLP class
mlp = MLP_stax(loss_function=loss_fn)
hidden_layers = myconfig.getlistint("training.mlp", "hidden_layers")
net_init, net_apply = mlp.init_mlp(n_targets, hidden_layers)

# initialize the optimizer
net_params_init, opt_init, opt_update, get_params = mlp.init_optimizer(
    step_size,
    n_features,
    seed=seed,
)

_ = mlp.train_mlp(
    input_data,
    target_data,
    num_epochs=n_epochs,
    batch_size=batch_size,
    num_batches=n_batches,
    timeit=True,
)

print("Done.\n")

"""Save results"""
print("Saving model to data")

savedir = config["training.paths"]["savedir_model"]
save_base_name = config["training.paths"]["save_base_name_model"]

mlp.save_model(
    savedir=savedir,
    save_base_name=save_base_name,
    verbose=True,
)

print("Done.\n")
