"""
Default CCSHMF parameters
"""

from collections import OrderedDict, namedtuple

# Ytp model params
DEFAULT_YTP_PDICT = OrderedDict(
    ytp_ytp=0.06, ytp_x0=13.72, ytp_k=1.95, ytp_ylo=0.06, ytp_yhi=0.15
)
YTP_Params = namedtuple("Ytp_Params", DEFAULT_YTP_PDICT.keys())
DEFAULT_YTP_PARAMS = YTP_Params(**DEFAULT_YTP_PDICT)

# Ylo model params
DEFAULT_YLO_PDICT = OrderedDict(
    ylo_ytp=-0.98, ylo_x0=12.40, ylo_k=1.67, ylo_ylo=0.34, ylo_yhi=0.09
)
YLO_Params = namedtuple("Ylo_Params", DEFAULT_YLO_PDICT.keys())
DEFAULT_YLO_PARAMS = YLO_Params(**DEFAULT_YLO_PDICT)


DEFAULT_CCSHMF_PDICT = OrderedDict(
    ytp_params=DEFAULT_YTP_PARAMS, ylo_params=DEFAULT_YLO_PARAMS
)
CCSHMF_Params = namedtuple("CCSHMF_Params", DEFAULT_CCSHMF_PDICT.keys())
DEFAULT_CCSHMF_PARAMS = CCSHMF_Params(**DEFAULT_CCSHMF_PDICT)
