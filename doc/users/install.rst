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

Nitime supports Python_ 2.7 and 3.3/3.4/3.5, requiring also reasonably recent
versions of NumPy_ and SciPy_, as well as Matplotlib_ (In particular,
:mod:`Nitime` makes use of the :mod:`matplotlib.mlab` module for some
implementation of numerical algorithms)

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

Using the standard `pip` installation mechanism, you can install nitime by
issuing the following command in your terminal::

     pip install nitime

The source code of the latest release is also available to download at the
cheeseshop_, or in our Github repo release page_

.. _page: gh-download_

.. _cheeseshop: nitime-pypi_

If you want to download the source-code as it is being developed (pre-release),
follow the instructions here: :ref:`following-latest`

Or, if you just want to look at the current development, without using our
source version-control system, you can download it directly here_

.. _here: gh-archive_

You can also install nitime using conda_, by issuing the following commands::

    conda config --add channels conda-forge
    conda install nitime



Building from source
--------------------

The installation process is similar to other Python packages so it
will be familiar if you have Python experience. In addition to the previously
mentioned dependencies, you will need to have cython_ installed

Unpack the tarball and change into the source directory.  Once in the
source directory, you can build nitime using::

    python setup.py install

Or::

    sudo python setup.py install

.. include:: ../links_names.txt
