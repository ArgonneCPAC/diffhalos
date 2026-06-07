""" """

from diffsky.mass_functions import ccshmf_model

ytp_params = ccshmf_model.YTP_Params(
    ytp_ytp=-0.032,
    ytp_x0=12.622,
    ytp_k=0.985,
    ytp_ylo=-0.134,
    ytp_yhi=0.134,
)

ylo_params = ccshmf_model.YLO_Params(
    ylo_ytp=-1.079,
    ylo_x0=12.451,
    ylo_k=1.646,
    ylo_ylo=-0.355,
    ylo_yhi=0.033,
)


CCSHMF_PARAMS = ccshmf_model.DEFAULT_CCSHMF_PARAMS._replace(
    ytp_params=ytp_params, ylo_params=ylo_params
)
