======================================
 Release notes for nitime version 0.5
======================================

Summary of changes
------------------

Major changes introduced in version 0.5 of :mod:`nitime`:

#.  Python 3 support. Work led by Paul Ivanov and Ariel Rokem, with help from
 Thomas Kluyver, and Matthew Brett.

#.  Continuous integration testing with Travis. Work led by Ariel Rokem.

#.  Various fixes and robustifications from several contributors (see below). 

Contributors to this release
----------------------------

The following people contributed to this release:

* Ariel Rokem
* Dmitry Shachnev
* Eric Larson
* Mike Trumpis
* Paul Ivanov
* endolith

.. Note::

   This list was generated using::

   git log --pretty=format:"* %aN" rel/0.4... | sort | uniq

   Please let us know if you should appear on this list and do not, so that we
   can add your name in future release notes.


Detailed stats from the github repository
-----------------------------------------

GitHub stats for the last 730 days.  We closed a total of 40 issues, 17 pull
requests and 23 regular issues; this is the full list (generated with the
script `tools/github_stats.py`):

Pull Requests (17):

* :ghissue:`124`: Buildbot mpl
* :ghissue:`114`: This should help the buildbot on older platforms
* :ghissue:`122`: Mpl units patch
* :ghissue:`121`: RF: Remove the dependency on external 'six', by integrating that file in
* :ghissue:`119`: Python3 support!
* :ghissue:`120`: Pi=py3k
* :ghissue:`115`: BF: For complex signals, return both the negative and positive spectrum
* :ghissue:`112`: Mt fix ups
* :ghissue:`118`: FIX: Pass int
* :ghissue:`117`: NF: On the way to enabling travis ci.
* :ghissue:`111`: Use inheritance_diagram.py provided by Sphinx (>= 0.6)
* :ghissue:`110`: BF + TST: Robustification and testing of utility function.
* :ghissue:`108`: Doc timeseries
* :ghissue:`109`: Spectra for multi-dimensional time-series
* :ghissue:`107`: DOC: fix parameter rendering for timeseries
* :ghissue:`106`: Fix rst definition list formatting 
* :ghissue:`105`: FIX: Kmax wrong, BW = bandwidth

Issues (23):

* :ghissue:`116`: Refer to github more prominently on webpage
* :ghissue:`124`: Buildbot mpl
* :ghissue:`114`: This should help the buildbot on older platforms
* :ghissue:`123`: Memory error of  GrangerAnalyzer
* :ghissue:`122`: Mpl units patch
* :ghissue:`121`: RF: Remove the dependency on external 'six', by integrating that file in
* :ghissue:`120`: Pi=py3k
* :ghissue:`119`: Python3 support!
* :ghissue:`115`: BF: For complex signals, return both the negative and positive spectrum
* :ghissue:`112`: Mt fix ups
* :ghissue:`118`: FIX: Pass int
* :ghissue:`117`: NF: On the way to enabling travis ci.
* :ghissue:`113`: Race condition provoked in TimeArray
* :ghissue:`111`: Use inheritance_diagram.py provided by Sphinx (>= 0.6)
* :ghissue:`110`: BF + TST: Robustification and testing of utility function.
* :ghissue:`108`: Doc timeseries
* :ghissue:`109`: Spectra for multi-dimensional time-series
* :ghissue:`107`: DOC: fix parameter rendering for timeseries
* :ghissue:`106`: Fix rst definition list formatting 
* :ghissue:`105`: FIX: Kmax wrong, BW = bandwidth
* :ghissue:`30`: Make default behavior for fmri.io.time_series_from_file
* :ghissue:`84`: Note on examples
* :ghissue:`93`: TimeArray .prod is borked (because of overflow?)
