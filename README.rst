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
The `diffhalos_quickstart.ipynb` notebook illustrates how to use some of the core functions of diffhalos. In particular, it demonstrates 
how to generate halo and subhalo lightcones using the Monte-Carlo and quasi-Monte Carlo versions of the generator. In addition, 
it shows that the lightcone outputs agree well with the exact theory predictions from the halo and subhalo mass functions in diffhalos. 

The `diffhalos_mah.ipynb` notebook goes over the basics of using the `DiffmahNet` model in `Diffhalos` to generate mass assembly 
histories (MAHs) of populations of dark matter halos. It also demonstrates how the MAH of halos is part of the standard output 
from lightcone generation.