======================================
 Release notes for nitime version 0.4
======================================

Summary of changes
------------------

Major changes introduced in version 0.4 of :mod:`nitime`:

#.  :class:`LazyImports <LazyImport>`: Imports of modules are delayed until they are actually
     used. Work led by Paul Ivanov

#. :class:`TimeArray` math: Mathematical operations such as multiplication/division, as
     well as min/max/mean/sum are now implemented for the TimeArray class. Work led
     by Paul Ivanov.

#. Replace numpy FFT with scipy FFT. This should improve performance. Work
    instigated and led by Alex Gramfort.

#. Scipy > 0.10 compatibility: Changes to recent versions of scipy have caused
    import of some modules of nitime to break. This version should have fixed this
    issue.

Contributors to this release
----------------------------

The following people contributed to this release:

* Alexandre Gramfort
* Ariel Rokem
* endolith
* Paul Ivanov
* Sergey Karayev
* Yaroslav Halchenko


.. Note::

   This list was generated using::

   git log  --format="%aN"  rel/0.3...  | sort | uniq

   Please let us know if you should appear on this list and do not, so that we
   can add your name in future release notes.


Detailed stats from the github repository
-----------------------------------------

Github stats for the last  XXX days.
We closed a total of XXX issues,XXX pull requests and XXX regular issues; this
is the full list (generated with the script  `tools/github_stats.py`):

* Alexandre Gramfort
* Ariel Rokem
* Paul Ivanov
* Sergey Karayev
* Yaroslav Halchenko
* endolith


