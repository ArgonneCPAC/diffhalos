diffhalos
============

Diffhalos is a generative model of cosmological lightcones of dark matter halos, subhalos, and their merger trees. 
This library is written in JAX, and makes differentiable predictions for halo populations in a lightcone, 
or at a single redshift.

Documentation
-------------
Online documentation is available at `diffhalos.readthedocs.io <https://diffhalos.readthedocs.io/en/latest/>`_

Installation
------------
To install diffhalos into your environment from the source code::

    $ cd /path/to/root/diffhalos
    $ pip install .

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
The :code:`diffhalos_quickstart.ipynb` notebook illustrates how to use some of the core functions of diffhalos. In particular, it demonstrates 
how to generate halo and subhalo lightcones using the Monte-Carlo and quasi-Monte Carlo versions of the generator. In addition, 
it shows that the lightcone outputs agree well with the exact theory predictions from the halo and subhalo mass functions in Diffhalos. 

In the notebook :code:`diffhalos_mah.ipynb` we go over the basics of using the DiffmahNet model in Diffhalos to generate mass assembly 
histories (MAHs) of populations of dark matter halos. It also demonstrates how the MAH of synthetic halos is part of the standard output 
in the lightcones.

We demonstrate and explain in more detail the fundamentals of using the Monte Carlo and quasi-Monte Carlo generators, and discuss their differences, 
in the notebook :code:`diffhalos_mc_vs_qmc.ipynb`. In there, we cover the basics of how to use each version for host halos, 
and we talk about how they differ in their approch to lightcone generation.