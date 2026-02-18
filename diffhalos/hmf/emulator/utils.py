"""Useful utilities"""

import datetime

__all__ = (
    "format_time",
    "MyConfigParser",
)


def format_time(delta_t):
    """
    Helper function to format the time
    elapsed from seconds in hours:minutes:seconds
    """
    ans = str(datetime.timedelta(seconds=delta_t))
    h, m, s = ans.split(":")
    if h != "0":
        out = "{}h {}m {}s".format(h, m, s)
    elif m != "00":
        out = "{}m {}s".format(m, s)
    else:
        out = "{}s".format(s)
    return out


class MyConfigParser:
    def __init__(self, config):
        """
        Class holding methods that make it easy to
        read in things like lists using Python's ConfigParser
        """
        self.config = config

    def getlist(self, section, option, lst_sep=","):
        lst = self.config.get(section, option).split(lst_sep)
        return lst

    def getlistint(self, section, option):
        lst = self.getlist(section, option)
        return [int(x) for x in lst]

    def getlistfloat(self, section, option):
        lst = self.getlist(section, option)
        return [float(x) for x in lst]
