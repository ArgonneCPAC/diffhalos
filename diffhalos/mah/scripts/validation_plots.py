import pathlib
import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import corner

import diffmahnet
from diffmahnet import datatools

TRAIN_DATA_DIR = pathlib.Path(
    "/lcrc/project/halotools/diffmahpop_data/NM_12_NT_9_ISTART_0_IEND_576/"
)
SAVE_DIR = pathlib.Path("./data/")
PLOT_DIR = pathlib.Path("./plots/")
SAMPLE_FRAC = 1.0


def plot_mah_hists(
    flow,
    data,
    data_desc=None,
    flow_desc=None,
    tfrac=1.0,
    tobs_ranges=None,
    mobs_ranges=None,
    title="",
    randkey=jax.random.key(0),
):
    data_desc = data_desc + " data" if data_desc is not None else "Data"
    flow_desc = flow_desc + " flow samples" if flow_desc is not None else ""
    if mobs_ranges is None:
        mobs_ranges = [(11.45, 11.75), (12.3, 12.7), (13.15, 13.55), (14.2, 14.57)]
    if tobs_ranges is None:
        tobs_ranges = [(3, 4), (6, 7), (9, 10), (13.7, 14.1)]
    key1, key2 = jax.random.split(randkey, 2)

    mah_params = data.diffmahparams
    flow_mah_params = flow.sample(data.u, randkey=key1, asparams=True)

    tgrid = diffmahnet.gen_time_grids(key2, data.u[:, 1])
    tfrac_grid = tgrid / data.u[:, 1, None]
    tgrid_ind = jnp.minimum(
        jax.vmap(jnp.searchsorted, in_axes=(0, None))(tfrac_grid, tfrac),
        tfrac_grid.shape[1] - 1,
    )
    log_mah = diffmahnet.log_mah_kern(mah_params, tgrid, data.logt0)[
        np.arange(tgrid.shape[0]), tgrid_ind
    ]
    flow_log_mah = diffmahnet.log_mah_kern(flow_mah_params, tgrid, data.logt0)[
        np.arange(tgrid.shape[0]), tgrid_ind
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for i_tobs, ax in tqdm.tqdm(enumerate(axes.ravel()), leave=False):
        tobs_min, tobs_max = tobs_ranges[i_tobs]
        tobs_cut = (tobs_min < data.u[:, 1]) & (data.u[:, 1] < tobs_max)
        for i_mobs in tqdm.trange(len(mobs_ranges), leave=False):
            mobs_min, mobs_max = mobs_ranges[i_mobs]
            mobs_cut = (mobs_min < data.u[:, 0]) & (data.u[:, 0] < mobs_max)

            cut = tobs_cut & mobs_cut
            flow_hist_dat = flow_log_mah[cut]
            hist_dat = log_mah[cut]
            all_dat = np.concatenate([flow_hist_dat, hist_dat])
            if len(all_dat):
                mean = all_dat.mean()
                bins = np.linspace(mean - 3, mean + 3, 70)
            else:
                bins = 70
            color = f"C{i_mobs}"
            ax.hist(
                flow_hist_dat,
                bins=bins,
                linewidth=2,
                color=color,
                histtype="step",
                density=True,
            )
            ax.hist(
                hist_dat,
                bins=bins,
                linestyle="--",
                color=color,
                histtype="step",
                density=True,
            )
        ax.hist([], alpha=0, label=f"t_obs~{np.mean(tobs_ranges[i_tobs]):.1f}")
    ax = axes.ravel()[0]

    for i_mobs in range(len(mobs_ranges)):
        color = f"C{i_mobs}"
        m = np.mean(mobs_ranges[i_mobs])
        if flow_desc:
            label = flow_desc + f" (M_obs~{m:.1f})"
        else:
            label = f"M_obs~{m:.1f}"
        ax.hist([], linewidth=2, color=color, histtype="step", label=label)
    ax.hist([], linestyle="--", color="k", histtype="step", label=data_desc)
    for ax in axes.ravel():
        ax.set_xlim(left=9)
        if ax in axes[-1, :]:
            ax.set_xlabel("$\\rm M_h (t)$")
        ax.legend(frameon=False, fontsize=10)
    if title:
        fig.suptitle(title)
    plt.show()


def plot_mah_residual(
    flow, data, tobs_ranges=None, mobs_ranges=None, title="", randkey=jax.random.key(0)
):
    if mobs_ranges is None:
        mobs_ranges = [(11.45, 11.75), (12.3, 12.7), (13.15, 13.55), (14.2, 14.57)]
    if tobs_ranges is None:
        tobs_ranges = [(3, 4), (6, 7), (9, 10), (13.7, 14.1)]
    key1, key2 = jax.random.split(randkey, 2)

    mah_params = data.diffmahparams
    flow_mah_params = flow.sample(data.u, randkey=key1, asparams=True)

    tgrid = diffmahnet.gen_time_grids(key2, data.u[:, 1], n_tgrid=100)
    log_mah = diffmahnet.log_mah_kern(mah_params, tgrid, data.logt0)
    flow_log_mah = diffmahnet.log_mah_kern(flow_mah_params, tgrid, data.logt0)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=True)
    cmap = plt.matplotlib.colormaps["rainbow"]
    colors = cmap(np.linspace(0, 1, len(mobs_ranges)))
    for i_tobs, ax in enumerate(axes.ravel()):
        tobs_min, tobs_max = tobs_ranges[i_tobs]
        tobs_cut = (tobs_min < data.u[:, 1]) & (data.u[:, 1] < tobs_max)
        ax.axhline(0, color="k", ls="--")
        for i_mobs in range(len(mobs_ranges)):
            mobs_min, mobs_max = mobs_ranges[i_mobs]
            mobs_cut = (mobs_min < data.u[:, 0]) & (data.u[:, 0] < mobs_max)

            cut = tobs_cut & mobs_cut
            if np.any(cut):
                flow_mean = np.mean(flow_log_mah[cut], axis=0)
                dat_mean = np.mean(log_mah[cut], axis=0)
                tgrid_mean = np.mean(tgrid[cut], axis=0)
                color = colors[i_mobs]
                ax.plot(tgrid_mean, flow_mean - dat_mean, linewidth=2, color=color)
        ax.plot([], [], alpha=0, label=f"t_obs~{np.mean(tobs_ranges[i_tobs]):.1f}")

    for i_mobs in range(len(mobs_ranges)):
        color = colors[i_mobs]
        m = np.mean(mobs_ranges[i_mobs])
        ax.plot([], [], linewidth=2, color=color, label=f"M_obs~{m:.1f}")
    for ax in axes.ravel():
        # ax.set_xlim(left=9)
        if ax in axes[:, 0]:
            ax.set_ylabel("$\\rm\\Delta\\langle\\log M_h (t)\\rangle$")
        if ax in axes[-1, :]:
            ax.set_xlabel("t")
        ax.legend(frameon=False, fontsize=10)
    if title:
        fig.suptitle(title)
    plt.show()


def plot_mah_corner(
    flow, data, data_desc=None, flow_desc=None, randkey=jax.random.key(0)
):
    data_desc = data_desc + " data" if data_desc is not None else "Data"
    flow_desc = flow_desc + " flow samples" if flow_desc is not None else "Flow samples"
    key1, key2 = jax.random.split(randkey, 2)

    mah_params = data.diffmahparams
    flow_mah_params = flow.sample(data.u, randkey=key1, asparams=True)

    tgrid = diffmahnet.gen_time_grids(key2, data.u[:, 1])
    log_mah = diffmahnet.log_mah_kern(mah_params, tgrid, data.logt0)
    flow_log_mah = diffmahnet.log_mah_kern(flow_mah_params, tgrid, data.logt0)

    # Dimension indices correspond to: (object, variable, time snapshot)
    broadcasted_u = np.tile(data.u[:, :, None], (1, 1, tgrid.shape[-1]))
    combined_data = np.array(
        [
            broadcasted_u[:, 0, :].flatten(),
            broadcasted_u[:, 1, :].flatten(),
            tgrid.flatten(),
            log_mah.flatten(),
        ]
    ).T
    combined_flow = np.array(
        [
            broadcasted_u[:, 0, :].flatten(),
            broadcasted_u[:, 1, :].flatten(),
            tgrid.flatten(),
            flow_log_mah.flatten(),
        ]
    ).T

    labels = ["logM_obs", "t_obs", "t", "logMAH"]
    hist_kwargs = {"density": True}
    ranges = [1.0, 1.0, 1.0, (5, 15)]
    fig = corner.corner(
        combined_flow,
        color="C2",
        plot_datapoints=False,
        hist_kwargs={**hist_kwargs, "color": "C2"},
        labels=labels,
        range=ranges,
        plot_density=False,
    )
    corner.corner(
        combined_data,
        fig=fig,
        color="C1",
        plot_datapoints=False,
        hist_kwargs={**hist_kwargs, "color": "C1"},
        labels=labels,
        range=ranges,
        plot_density=False,
    )
    fig.axes[1].text(0, 0.1, data_desc, color="C1")
    fig.axes[1].text(0, 0, flow_desc, color="C2")
    plt.show()


parser = argparse.ArgumentParser(description="Plot DiffMahNet validation plots.")
parser.add_argument("SAVE_FILENAME", help="Filename the trained model is saved as.")
parser.add_argument(
    "PLOT_DIR", help=f"Directory to save the plots, relative to {PLOT_DIR}"
)
parser.add_argument(
    "--save_dir",
    type=str,
    default=SAVE_DIR,
    help="Directory the trained model is saved in.",
)
parser.add_argument(
    "--train_data_dir",
    type=str,
    default=TRAIN_DATA_DIR,
    help="Directory containing the training data.",
)
parser.add_argument("--sats", action="store_true")
parser.add_argument("--not-test", action="store_true")
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
    args = parser.parse_args()
    save_dir = pathlib.Path(args.save_dir)
    save_filename = args.SAVE_FILENAME
    train_data_dir = pathlib.Path(args.train_data_dir)
    is_cens = not args.sats
    is_test = not args.not_test
    plot_dir = PLOT_DIR / args.PLOT_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)
    sample_frac = args.sample_frac

    key = jax.random.key(args.seed)
    key1, *keys = jax.random.split(key, num=5)

    flow = diffmahnet.DiffMahFlow.load(save_dir / save_filename)
    data = datatools.DataHolder(
        train_data_dir,
        is_test=is_test,
        is_cens=is_cens,
        sample_frac=sample_frac,
        randkey=key1,
    )

    if is_test:
        tobs_ranges = [(1.5, 4), (4, 7), (7.5, 8.5), (10, 12)]
        mobs_ranges = [(10.8, 11.33), (11.6, 12.5), (12.5, 13.0), (14.0, 14.8)]
    else:
        tobs_ranges = None
        mobs_ranges = None

    plot_mah_hists(
        flow,
        data,
        title="t = t_obs",
        tobs_ranges=tobs_ranges,
        mobs_ranges=mobs_ranges,
        randkey=keys[0],
    )
    plt.savefig(plot_dir / "mah_hist_t1p0.png", bbox_inches="tight")
    plot_mah_hists(
        flow,
        data,
        title="t = 0.6 * t_obs",
        tfrac=0.6,
        tobs_ranges=tobs_ranges,
        mobs_ranges=mobs_ranges,
        randkey=keys[1],
    )
    plt.savefig(plot_dir / "mah_hist_t0p6.png", bbox_inches="tight")
    plot_mah_hists(
        flow,
        data,
        title="t = 0.3 * t_obs",
        tfrac=0.3,
        tobs_ranges=tobs_ranges,
        mobs_ranges=mobs_ranges,
        randkey=keys[2],
    )
    plt.savefig(plot_dir / "mah_hist_t0p3.png", bbox_inches="tight")

    plot_mah_residual(
        flow, data, tobs_ranges=tobs_ranges, mobs_ranges=mobs_ranges, randkey=keys[3]
    )
    plt.savefig(plot_dir / "mah_residual.png", bbox_inches="tight")
