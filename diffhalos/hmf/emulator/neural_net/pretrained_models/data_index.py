"""
Training data storage info.
Best practice is to store the models in a local folder
and set the environment variable that points to the
directory storing all the required model data,
as shown below.
"""

import pathlib

PATH_TO_HERE = str(pathlib.Path(__file__).parent)

"""
To temporary set the environment path, run the following:
    export DIFFHMFEMU_MODEL_DATA="path_to_data_directory"
The default environment variable name is the one below.
"""
ENVIRON_VAR = "DIFFHMFEMU_MODEL_DATA"
