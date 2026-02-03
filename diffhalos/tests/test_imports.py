""""""


def test_top_level_imports():
    """Enforce all key user-facing functions are importable directly from diffhalos"""
    from .. import redshift_mass_grid  # noqa
    from .. import weighted_lc  # noqa
