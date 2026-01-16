import jax
import pathlib
import glob

import numpy as np
import jax.numpy as jnp

from . import diffmahnet

from diffmah import DEFAULT_MAH_PARAMS
from diffmah.diffmah_kernels import get_unbounded_mah_params, get_bounded_mah_params

DEFAULT_MAH_UPARAMS = get_unbounded_mah_params(DEFAULT_MAH_PARAMS)


def load_all_tdata(path, is_test: bool | str = False, is_cens=True):
    # Parse available training data files
    tdata_files = glob.glob(str(pathlib.Path(path) / "*"))
    filenames = [x.split("/")[-1] for x in tdata_files]
    lgm_vals = np.array([float(x.split("_")[1]) for x in filenames])
    t_vals = np.array([float(x.split("_")[3]) for x in filenames])
    is_cens_vals = np.array([x.split(".")[-2] == "cens" for x in filenames])
    fileinfo = list(
        zip(tdata_files, lgm_vals.tolist(), t_vals.tolist(), is_cens_vals.tolist())
    )
    cen_file_inds = np.where(is_cens_vals)[0]
    sat_file_inds = np.where(~is_cens_vals)[0]

    # Load data
    test_train_file_split = 80  # about 25:75 test-train split ratio
    if is_test == "both":
        test_train_file_split = None
    inds = cen_file_inds if is_cens else sat_file_inds
    test_train_slice = slice(None, test_train_file_split)
    if is_test:
        test_train_slice = slice(test_train_file_split, None)
    inds = inds[test_train_slice]

    x = []  # PDF variables
    u = []  # conditional variables
    for i in inds:
        filename, lgm, t, is_cens_val = fileinfo[i]
        assert is_cens == is_cens_val
        x.append(np.load(filename))
        u.append(np.tile(np.array([[lgm, t]]), (x[-1].shape[0], 1)))

    x = jnp.concatenate(x, axis=0)
    u = jnp.concatenate(u, axis=0)

    # Transfrorm x parameters from bounded to unbounded space
    x_unbound = jnp.array([*get_unbounded_mah_params(DEFAULT_MAH_PARAMS._make(x.T))]).T

    isfinite = np.all((jnp.isfinite(x_unbound)), axis=1)
    return x_unbound[isfinite], u[isfinite]


class DataHolder:
    """Holds data for training and testing the flow"""

    def __init__(
        self,
        path,
        is_test: bool | str = False,
        is_cens=True,
        t0=13.8,
        sample_frac=1.0,
        randkey=None,
    ):
        if randkey is None:
            randkey = jax.random.key(0)
        self.logt0 = np.log10(t0)
        self.x, self.u = load_all_tdata(path, is_test=is_test, is_cens=is_cens)
        if sample_frac < 1:
            full_size = self.x.shape[0]
            sample = jax.random.choice(
                randkey, full_size, (int(sample_frac * full_size),), replace=False
            )
            self.x = self.x[sample]
            self.u = self.u[sample]
        self.m_obs, self.t_obs = self.u.T
        self.x_dim = self.x.shape[1]
        self.cond_dim = self.u.shape[1]

        self.scaler = diffmahnet.Scaler.compute(self.x, self.u)
        self.x_scaler = self.scaler.x_scaler
        self.u_scaler = self.scaler.u_scaler

        self.diffmahparams = get_bounded_mah_params(DEFAULT_MAH_UPARAMS._make(self.x.T))

    def get_tgrid_and_log_mah(self, randkey, t_min=0.1, n_tgrid=20):
        randkey = jax.random.split(randkey, 2)[1]
        mah_params = get_bounded_mah_params(DEFAULT_MAH_UPARAMS._make(self.x.T))

        tgrid = diffmahnet.gen_time_grids(
            randkey, self.t_obs, t_min=t_min, n_tgrid=n_tgrid
        )
        log_mah = diffmahnet.log_mah_kern(mah_params, tgrid, self.logt0)
        return tgrid, log_mah
