"""Useful namedtuple utilities"""

from collections import namedtuple

__all__ = ("add_field_to_namedtuple",)


def add_field_to_namedtuple(
    ntup,
    new_fields,
    new_data,
):
    """
    Parameters
    ----------
    ntup: namedtuple
        namedtuple to modify by adding fields

    new_fields_name: tuple(str)
        names of the new fields

    new_fields_values: tuple(data)
        data of the new fields

    Returns
    -------
    ntup_new: namedtuple
        new namedtuple with added fields
    """
    _name = type(ntup).__name__
    _fields = tuple(ntup._fields) + tuple(new_fields)

    ntup_new_dict = ntup._asdict()
    for i, _field in enumerate(new_fields):
        ntup_new_dict[_field] = new_data[i]

    ntup_new = namedtuple(_name, _fields)(**ntup_new_dict)

    return ntup_new
