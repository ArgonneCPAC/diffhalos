# """ """

# import numpy as np
# from jax import random as jran

# from .. import mc_lightcone_halos as mclh
# from .. import mc_lightcone_subhalos as mclsh
# from ...calibrations.ccshmf_cal import DEFAULT_CCSHMF_PARAMS  # noqa
# from ...ccshmf import mc_subs


# def test_mc_weighted_subhalo_lightcone_behaves_as_expected():
#     """
#     Enforce mc_lightcone_host_halo_diffmah
#     returns reasonable results when passed
#     alternative halo mass function parameters
#     """

#     ran_key = jran.key(0)
#     z_max = 0.5
#     z_min = z_max - 0.05
#     lgmp_max = 15.0
#     sky_area_degsq = 1.0
#     num_halos = 500

#     n_tests = 5
#     lgmp_min_arr = np.linspace(11.0, 13.0, n_tests)
#     for lgmp_min in lgmp_min_arr:
#         test_key, ran_key = jran.split(ran_key, 2)

#         cenpop = mclh.mc_weighted_halo_lightcone(
#             test_key,
#             num_halos,
#             z_min,
#             z_max,
#             lgmp_min,
#             lgmp_max,
#             sky_area_degsq,
#         )

#         satpop = mclsh.mc_weighted_subhalo_lightcone(
#             cenpop,
#             lgmp_min,
#             ccshmf_params=DEFAULT_CCSHMF_PARAMS,
#         )

#         assert "nsubhalos" in satpop
#         assert "logmu_subs" in satpop

#         assert np.all(np.isfinite(satpop["nsubhalos"]))
#         assert satpop["nsubhalos"].shape == (
#             satpop["logmp_obs"].size,
#             mclsh.N_LGMU_TABLE,
#         )

#         assert np.all(np.isfinite(satpop["logmu_subs"]))
#         assert satpop["logmu_subs"].shape == (
#             satpop["logmp_obs"].size,
#             mclsh.N_LGMU_TABLE,
#         )


# def test_mc_weighted_subhalo_lightcone_agrees_with_mc_subhalopop():

#     ran_key = jran.key(0)
#     num_halos = 500

#     lgmhost_arr = np.linspace(11.0, 13.0, num_halos)
#     cenpop = {"logmp_obs": lgmhost_arr}

#     n_tests = 5
#     lgmp_min_arr = np.linspace(9.0, 10.0, n_tests)
#     for lgmp_min in lgmp_min_arr:
#         satpop = mclsh.mc_weighted_subhalo_lightcone(
#             cenpop,
#             lgmp_min,
#             ccshmf_params=DEFAULT_CCSHMF_PARAMS,
#         )

#         mc_lg_mu = mc_subs.generate_subhalopop(
#             ran_key,
#             lgmhost_arr,
#             lgmp_min,
#             ccshmf_params=DEFAULT_CCSHMF_PARAMS,
#         )[0]

#         assert np.allclose(
#             mc_lg_mu.size,
#             satpop["nsubhalos"].sum(),
#             rtol=0.1,
#         )
