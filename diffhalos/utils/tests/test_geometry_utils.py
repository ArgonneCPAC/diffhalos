import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY

from ..geometry_utils import (
    compute_volume_from_sky_area,
    spherical_shell_comoving_volume,
)


def test_spherical_shell_comoving_volume():
    z_grid = np.linspace(1, 2, 25)
    vol_shell_grid = spherical_shell_comoving_volume(z_grid, DEFAULT_COSMOLOGY)
    assert vol_shell_grid.shape == z_grid.shape
    assert np.all(np.isfinite(vol_shell_grid))
    assert np.all(vol_shell_grid > 0)


def test_compute_volume_from_sky_area():

    redshift = np.linspace(0.1, 3.0, 10)
    sky_area_degsq = 10.0

    vol_com_mpc = compute_volume_from_sky_area(
        redshift,
        sky_area_degsq,
        DEFAULT_COSMOLOGY,
    )

    assert np.all(np.isfinite(vol_com_mpc))
    assert np.all(vol_com_mpc > 0.0)
