""" """

from collections import namedtuple

from ..namedtuple_utils import add_field_to_namedtuple


def test_add_field_to_namedtuple():

    ntup_init = namedtuple("ntup", ("a", "b"))
    ntup = ntup_init(a=1, b=2)

    new_fields = ("c", "d")
    new_data = (3, [4, 5])

    ntup_new = add_field_to_namedtuple(ntup, new_fields, new_data)

    all_fields = ("a", "b", "c", "d")

    assert type(ntup_new).__name__ == type(ntup).__name__
    for _field in all_fields:
        assert _field in ntup_new._fields
    assert ntup_new.a == ntup.a
    assert ntup_new.b == ntup.b
    assert ntup_new.c == 3
    assert ntup_new.d == [4, 5]
