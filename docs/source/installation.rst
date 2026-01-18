Installation
===============

Installing the latest release
-----------------------------

The latest release of ``diffhalos`` is available for installation with conda-forge::

       conda install -c conda-forge diffhalos

or with pip::

       pip install diffhalos


Installing the main branch
---------------------------------

For the most up-to-date version diffhalos that includes unreleased features,
you can install the main branch of the code::

       git clone https://github.com/ArgonneCPAC/diffhalos.git
       cd diffhalos
       pip install .


Dependencies
~~~~~~~~~~~~
Diffhalos dependencies should be handled automatically if you install with conda-forge
or pip, but if you install manually from source then you may need to install
`numpy <https://numpy.org/>`__,
`jax <https://jax.readthedocs.io/en/latest/>`__,
`diffmah <https://diffmah.readthedocs.io/en/latest/>`__,
and other dependencies in the diffstuff stack that appear in
`diffhalos/requirements.txt <https://github.com/ArgonneCPAC/diffhalos/blob/main/requirements.txt>`__.
