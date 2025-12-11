# flake8: noqa: E402
from jax import config

config.update("jax_enable_x64", True)

import pathlib
import argparse

import jax
import jax.numpy as jnp
import equinox as eqx

from diffopt import kdescent
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
NUM_EVAL_KERNELS = 20
NUM_EVAL_FOURIER_POSITIONS = 0
LEARNING_RATE = 1e-4


class KDescentLoss:
    """
    Custom loss function to fit flowjax model
    """

    def __init__(
        self,
        train_data,
        sample_size=None,
        randkey=None,
        num_eval_kernels=20,
        num_eval_fourier_positions=0,
    ):
        randkey = jax.random.key(0) if randkey is None else randkey
        # t0 = 13.8
        # self.logt0 = np.log10(t0)
        self.logt0 = train_data.logt0
        self.sample_size = sample_size
        self.xscaler = train_data.x_scaler
        self.uscaler = train_data.u_scaler

        self.tgrids, self.log_mah = train_data.get_tgrid_and_log_mah(randkey)
        self.m_obs = train_data.m_obs
        self.t_obs = train_data.t_obs
        assert self.log_mah.ndim == self.tgrids.ndim == 2
        assert self.m_obs.ndim == self.t_obs.ndim == 1
        assert (
            self.log_mah.shape[0]
            == self.m_obs.shape[0]
            == self.t_obs.shape[0]
            == self.tgrids.shape[0]
        )
        self.condition = jnp.array([self.m_obs, self.t_obs]).T

        # Combine m and t with condition (m_obs, t_obs), since we always have
        # an equivalent sampling of the conditional variables
        # and this saves us from having to generate many separate
        # KCalc instances at different conditional value bins
        self.training_combined = (
            jnp.array(
                [
                    self.log_mah,
                    self.tgrids,
                    self.tile(self.m_obs),
                    self.tile(self.t_obs),
                ]
            )
            .reshape((4, -1))
            .T
        )
        self.kde = kdescent.KCalc(
            self.training_combined,
            num_eval_kernels=num_eval_kernels,
            num_eval_fourier_positions=num_eval_fourier_positions,
        )

    def tile(self, arr):
        return jnp.tile(arr[..., None], (1, self.tgrids.shape[1]))

    @eqx.filter_jit
    def __call__(self, diffmahflow, randkey):
        """Compute the loss using kdescent"""
        key0, key1, key2, key3 = jax.random.split(randkey, 4)
        if self.sample_size is None:
            tsamp = slice(None)
        else:
            tsamp = jax.random.choice(
                key0,
                self.training_combined.shape[0],
                (self.sample_size,),
                replace=False,
            )
        mah_params = diffmahflow.sample(
            self.condition[tsamp], randkey=key1, asparams=True
        )
        log_mah = diffmahnet.log_mah_kern(mah_params, self.tgrids[tsamp], self.logt0)
        model_combined = (
            jnp.array(
                [
                    log_mah,
                    self.tgrids[tsamp],
                    self.tile(self.m_obs[tsamp]),
                    self.tile(self.t_obs[tsamp]),
                ]
            )
            .reshape((4, -1))
            .T
        )

        if self.kde.num_eval_fourier_positions:
            counts_model, counts_truth = self.kde.compare_fourier_counts(
                key2, model_combined
            )
            ecf_model = counts_model / model_combined.shape[0]
            ecf_truth = counts_truth / self.training_combined.shape[0]
            loss = jnp.sum(jnp.abs(ecf_model - ecf_truth) ** 2)
        else:
            loss = 0.0

        counts_model, counts_truth = self.kde.compare_kde_counts(key3, model_combined)
        pdf_model = counts_model / model_combined.shape[0]
        pdf_truth = counts_truth / self.training_combined.shape[0]
        loss += jnp.sum((pdf_model - pdf_truth) ** 2)

        # Optionally divide by total number of kernels to get MSE loss
        # loss /= (self.kde.num_eval_kernels + self.kde.num_eval_fourier_positions)
        jax.debug.print("loss = {loss}", loss=loss)

        return loss


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
parser.add_argument("--steps", type=int, default=100, help="Number of adam iterations.")
parser.add_argument(
    "--learning-rate",
    type=float,
    default=LEARNING_RATE,
    help="Initial adam learning rate.",
)
parser.add_argument(
    "--num-eval-kernels",
    type=int,
    default=NUM_EVAL_KERNELS,
    help="Number of kdescent kernels.",
)
parser.add_argument(
    "--num-eval-fourier-positions",
    type=int,
    default=NUM_EVAL_FOURIER_POSITIONS,
    help="Number of kdescent fourier kernels.",
)
parser.add_argument(
    "--sample-frac",
    type=float,
    default=SAMPLE_FRAC,
    help="Fraction of training data to load.",
)
parser.add_argument(
    "--seed", type=int, default=0, help="Random seed for reproducibility."
)
parser.add_argument(
    "--plot-loss-curve",
    action="store_true",
    help="Plot the loss curve during training.",
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
    steps = args.steps
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
    if steps > 0:
        loss_func = KDescentLoss(
            train_data,
            num_eval_kernels=args.num_eval_kernels,
            num_eval_fourier_positions=args.num_eval_fourier_positions,
        )
        params, losses = flow.adam_fit(
            loss_func, randkey=key2, nsteps=steps, learning_rate=args.learning_rate
        )
        if args.plot_loss_curve:
            import matplotlib.pyplot as plt

            plt.semilogy(losses)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plot_filename = save_filename.removesuffix(".eqx") + ".png"
            plt.savefig(save_dir / plot_filename)
            plt.close()

    # Save the trained model
    flow.save(save_dir / save_filename)
