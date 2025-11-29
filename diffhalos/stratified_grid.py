""" """

from functools import partial

from diffsky.experimental.lc_utils import spherical_shell_comoving_volume
from diffsky.mass_functions import hmf_model, mc_hosts
from dsps.cosmology import DEFAULT_COSMOLOGY, flat_wcdm
from jax import grad
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

FULL_SKY_AREA = (4 * jnp.pi) * (180 / jnp.pi) ** 2
N_HMF_GRID = 500


_dn_dlgm_kern = jjit(grad(hmf_model._diff_hmf_grad_kern, argnums=1))


@jjit
def _dn_dm_dz_kern(lgm, z, hmf_params, cosmo_params):
    dn_dm_dv = _dn_dlgm_kern(hmf_params, lgm, z)
    dv_dz = flat_wcdm.differential_comoving_volume_at_z(z, *cosmo_params)
    dn_dm_dz = dn_dm_dv * dv_dz
    return dn_dm_dz


_A = (0, 0, None, None)
predict_dn_dlgm_dz = jjit(vmap(_dn_dm_dz_kern, in_axes=_A))


@partial(jjit, static_argnames=["n_per_dim"])
def stratified_xy_grid(n_per_dim, ran_key):
    """
    Stratified grid with noise

    Parameters
    ----------
    n_per_dim: int
        Number of points per dimension (total = n_per_dim^2)

    ran_key: jax.random.key(seed)

    Returns
    -------
    xy_grid : array, shape (n_per_dim^2, 2)
        0 <= x,y <= 1 for every grid element

    """
    grid_1d = (jnp.arange(n_per_dim) + 0.5) / n_per_dim

    x = jnp.repeat(grid_1d, n_per_dim)
    y = jnp.tile(grid_1d, n_per_dim)
    xy_grid = jnp.column_stack([x, y])

    uran = jran.uniform(ran_key, shape=xy_grid.shape)
    noise = uran / n_per_dim - 0.5 / n_per_dim

    return xy_grid + noise


@partial(jjit, static_argnames=["n_per_dim"])
def mc_halo_lightcone(
    n_per_dim,
    ran_key,
    z_min,
    z_max,
    lgmp_min,
    sky_area_degsq,
    cosmo_params=DEFAULT_COSMOLOGY,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    lgmp_max=17,
):
    # Set up a uniform grid in redshift
    z_grid = jnp.linspace(z_min, z_max, N_HMF_GRID)

    # Compute the comoving volume of a thin shell at each grid point
    fsky = sky_area_degsq / FULL_SKY_AREA
    vol_shell_grid_mpc = fsky * spherical_shell_comoving_volume(z_grid, cosmo_params)

    # At each grid point, compute <Nhalos> for the shell volume
    mean_nhalos_grid = mc_hosts._compute_nhalos_tot(
        hmf_params, lgmp_min, z_grid, vol_shell_grid_mpc
    )
    mean_nhalos_lgmp_max = mc_hosts._compute_nhalos_tot(
        hmf_params, lgmp_max, z_grid, vol_shell_grid_mpc
    )
    mean_nhalos_grid = mean_nhalos_grid - mean_nhalos_lgmp_max

    nhalos_tot = mean_nhalos_grid.sum()

    xy_grid = stratified_xy_grid(n_per_dim, ran_key)
    um, uz = xy_grid[:, 0], xy_grid[:, 1]

    lgm = lgmp_min + (lgmp_max - lgmp_min) * um
    z = z_min + (z_max - z_min) * uz

    _weights = predict_dn_dlgm_dz(lgm, z, hmf_params, cosmo_params)
    weights = _weights / _weights.sum()
    nhalos = nhalos_tot * weights

    return lgm, z, nhalos
