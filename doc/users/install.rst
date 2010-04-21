.. _install:

======================
 Download and install
======================

This page covers the necessary steps to install nitime.  Below is a
list of required dependencies, along with additional software
recommendations.

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

Strong Recommandations
~~~~~~~~~~~~~~~~~~~~~~

IPython_ 0.10
  Interactive python environment. This is necessary for the parallel
  components of the pipeline engine.

Sphinx_
  Required for building the documentation

Networkx_
  Used for some visualization functions

Getting the latest release
--------------------------

The code is here: http://github.com/fperez/nitime

You can grab a copy of the code by using the following command (assuming you
have git installed)::

	git clone git://github.com/fperez/nitime.git

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
