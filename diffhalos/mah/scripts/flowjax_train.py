import pathlib
import argparse

import jax
import diffmahnet
from diffmahnet import datatools

SAVE_DIR = pathlib.Path("./data/")
TRAIN_DATA_DIR = pathlib.Path(
    "/lcrc/project/halotools/diffmahpop_data/NM_12_NT_9_ISTART_0_IEND_576/"
)

NN_DEPTH = 2
NN_WIDTH = 50
FLOW_LAYERS = 8
SAMPLE_FRAC = 1.0


parser = argparse.ArgumentParser(
    description="Train a DiffMahNet normalizing flow model."
)
parser.add_argument("SAVE_FILENAME", help="Filename to save the trained model.")
parser.add_argument(
    "--save-dir",
    type=str,
    default=SAVE_DIR,
    help="Directory to save the trained model.",
)
parser.add_argument(
    "--train-data-dir",
    type=str,
    default=TRAIN_DATA_DIR,
    help="Directory containing the training data.",
)
parser.add_argument(
    "--initial-model",
    type=str,
    default=None,
    help="Optional filename of an initial model to load.",
)
parser.add_argument("--sats", action="store_true")
parser.add_argument(
    "--nn-depth", type=int, default=NN_DEPTH, help="Depth of the hidden neural network."
)
parser.add_argument(
    "--nn-width", type=int, default=NN_WIDTH, help="Width of the hidden neural network."
)
parser.add_argument(
    "--flow-layers", type=int, default=FLOW_LAYERS, help="Number of flow layers."
)
parser.add_argument(
    "--include-test", action="store_true", help="Include test data in the training set."
)
parser.add_argument(
    "--max-epochs", type=int, default=50, help="Number of training epochs."
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=5e-4,
    help="Learning rate for the built-in flowjax optimizer.",
)
parser.add_argument("--max-patience", type=float, default=10.0)
parser.add_argument(
    "--sample-frac",
    type=float,
    default=SAMPLE_FRAC,
    help="Fraction of training data to load.",
)
parser.add_argument(
    "--seed", type=int, default=0, help="Random seed for reproducibility."
)

if __name__ == "__main__":
    # Parse arguments
    args = parser.parse_args()
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_filename = args.SAVE_FILENAME
    train_data_dir = pathlib.Path(args.train_data_dir)
    is_cens = not args.sats
    nn_depth = args.nn_depth
    nn_width = args.nn_width
    flow_layers = args.flow_layers
    is_test = "both" if args.include_test else False
    initial_model = args.initial_model
    max_epochs = args.max_epochs
    sample_frac = args.sample_frac

    key = jax.random.key(args.seed)
    key1, key2 = jax.random.split(key)

    # Load training data and flow model
    train_data = datatools.DataHolder(
        train_data_dir,
        is_cens=is_cens,
        is_test=is_test,
        sample_frac=sample_frac,
        randkey=key1,
    )
    if initial_model is not None:
        initial_model = save_dir / initial_model
        flow = diffmahnet.DiffMahFlow.load(initial_model)
    else:
        flow = diffmahnet.DiffMahFlow(
            scaler=train_data.scaler,
            nn_depth=nn_depth,
            nn_width=nn_width,
            flow_layers=flow_layers,
        )
    print("Number of parameters =", flow.get_params().size)

    # Train the flow model
    if max_epochs > 0:
        print("Training data shapes:", train_data.x.shape, train_data.u.shape)
        flow.init_fit(
            train_data.x,
            train_data.u,
            randkey=key2,
            max_epochs=max_epochs,
            learning_rate=args.learning_rate,
            max_patience=args.max_patience,
        )

    # Save the trained model
    flow.save(save_dir / save_filename)
