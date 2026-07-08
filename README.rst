diffhalos
============

Installation
------------
To install diffhalos into your environment from the source code::

    $ cd /path/to/root/diffhalos
    $ pip install .


Documentation
-------------
Online documentation is available at `diffhalos.readthedocs.io <https://diffhalos.readthedocs.io/en/latest/>`_


Testing
-------
To run the suite of unit tests::

    $ cd /path/to/root/diffhalos
    $ pytest

To build html of test coverage::

    $ pytest -v --cov --cov-report html
    $ open htmlcov/index.html

Demo notebooks
--------------
The `diffhalos_quickstart.ipynb` notebook illustrates how to use some of the core functions of diffhalos. In particular,
it demonstrates how to generate halo and subhalo lightcones, while it also shows that the output from the Monte-Carlo
lightcone realizations agree with the exact theory predictions from the halo mass function and the conditional
subhalo mas function implemented in diffhalos.