.. _install:

======================
 Download and install
======================

This page covers the necessary steps to install nitime.  Below is a
list of required dependencies, and some additional recommended software, which
are dependencies for some particular functionality.

Dependencies
------------

Must Have
~~~~~~~~~

Python_ 2.5 or later

NumPy_ 1.3 or later

SciPy_ 0.7 or later
  Numpy and Scipy are high-level, optimized scientific computing libraries.

Matplotlib_
  Python plotting library. In particular, :mod:`Nitime` makes use of the
  :mod:`matplotlib.mlab` module for some implementation of numerical algorithms

Recommended/optional
~~~~~~~~~~~~~~~~~~~~

Sphinx_
  Required for building the documentation

Networkx_
  Used for some visualization functions; required in order to build the
  documentation.

Nibabel_
  Used for reading in data from fMRI data files; required in order to build the
  documentation.

Getting the latest release
--------------------------

If you have easy_install_ available on your system, nitime can be downloaded and
install by issuing::

    easy_install nitime

.. _easy_install: easy-install_

Otherwise, you can grab the latest release of the source-code at this page_ 

.. _page: gh-download_

Or, at the cheeseshop_

.. _cheeseshop: nitime-pypi_

If you want to download the source-code as it is being developed (pre-release),
follow the instructions here: :ref:`following-latest`

Or, if you just want to look at the current development, without using our
source version-control system, go here_

.. _here: gh-archive_


Building from source
--------------------

The installation process is similar to other Python packages so it
will be familiar if you have Python experience.

Unpack the tarball and change into the source directory.  Once in the
source directory, you can build nitime using::

    python setup.py install

Or::

    sudo python setup.py install

.. include:: ../links_names.txt
