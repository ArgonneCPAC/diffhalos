diffhalos
============

Installation
------------
To install diffhalos into your environment from the source code::

    $ cd /path/to/root/diffhalos
    $ pip install .


Documentation
------------
Online documentation is available at [diffhalos.readthedocs.io](https://diffhalos.readthedocs.io/en/latest/)


Testing
-------
To run the suite of unit tests::

    $ cd /path/to/root/diffhalos
    $ pytest

To build html of test coverage::

    $ pytest -v --cov --cov-report html
    $ open htmlcov/index.html

