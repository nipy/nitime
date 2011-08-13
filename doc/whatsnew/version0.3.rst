
======================================
 Release notes for nitime version 0.3
======================================

Summary of changes
------------------

Version 0.3 of nitime includes several additions and improvements: 

#. Testing: Test coverage of nitime has improved substantially. At this point,
   84 % of the code is executed when the test suite is executed. This includes
   a full 100% execution of all l.o.c. in most of the algorithms
   sub-module. Work led by Ariel Rokem.

#. Style and layout improvements: The layout of the algorithms and analysis
   sub-modules have been simplified and a large majority of the code-base has
   been modified to conform with PEP8 standards. Work led by Ariel Rokem, with
   additional thanks to Alex Gramfort for pushing for these changes and helping
   to bring them about.

#. Bug-fixes to the SNRAnalyzer: Several bugs in this module have been
   fixed. Work led by Ariel Rokem (who put these bugs in there in the first
   place...).  

#. MAR estimation algorithms: Extensive reworking of MAR estimation algorithms.
   Work led by Mike Trumpis. 

#. SeedCorrelationAnalyzer: This analyzer allows flexible correlation analysis
   in a few-to-many channel mode. Work led by Michael Waskom. 

#. GrangerAnalyzer: Following Mike Trumpis' work on MAR estimation, we have
   implemented an Analyzer for Granger 'causality' analysis. Work led by Ariel
   Rokem, Mike Trumpis and Fernando Perez 

#. Filtering: Implementation of zero phase-delay filtering, including IIR and
   FIR filter methods to the FilterAnalyzer. Work led by Ariel Rokem 

#. Several new examples, including examples of the usage of these new analysis
   methods. 

#. Epoch slicing: Additional work on TimeSeries objects, towards an
   implementation. This feature is still at an experimental stage at this
   point. Work led by Ariel Rokem, Fernando Perez, Killian Koepsell and Paul
   Ivanov.  
   


Contributors to this release
----------------------------

* Alexandre Gramfort
* Ariel Rokem
* Christopher Burns 
* Fernando Perez 
* Jarrod Millman 
* Killian Koepsell
* Michael Waskom 
* Mike Trumpis
* Paul Ivanov 
* Yaroslav Halchenko 

.. Note::

   This list was generated using::
   
       git log dev/0.3 HEAD --format='* %aN <%aE>' |sed 's/@/\-at\-/' | sed 's/<>//' | sort -u  

   Please let us know if you should appear on this list and do not, so that we
   can add your name in future release notes. 

       
Detailed stats from the github repository
-----------------------------------------

Github stats for the last  270 days.
We closed a total of 38 issues, 28 pull requests and 10 regular 
issues; this is the full list (generated with the script 
`tools/github_stats.py`):

Pull requests (28):

* `78 <https://github.com/nipy/nitime/issues/78>`_: Doctests
* `76 <https://github.com/nipy/nitime/issues/76>`_: Sphinx warnings
* `74 <https://github.com/nipy/nitime/issues/74>`_: BF: IIR filtering can do band-pass as well as low-pass and high-pass.
* `72 <https://github.com/nipy/nitime/issues/72>`_: ENH: Throw an informative warning when time-series is short for the NFFT.
* `75 <https://github.com/nipy/nitime/issues/75>`_: ENH: Default behavior for time_series_from_file.
* `71 <https://github.com/nipy/nitime/issues/71>`_: Granger analyzer
* `73 <https://github.com/nipy/nitime/issues/73>`_: Seed correlation analyzer
* `69 <https://github.com/nipy/nitime/issues/69>`_: BF: add back tril_indices from numpy 1.4, to support operation with older
* `70 <https://github.com/nipy/nitime/issues/70>`_: Ar latex
* `67 <https://github.com/nipy/nitime/issues/67>`_: Mar examples
* `66 <https://github.com/nipy/nitime/issues/66>`_: Test coverage
* `63 <https://github.com/nipy/nitime/issues/63>`_: Utils work
* `62 <https://github.com/nipy/nitime/issues/62>`_: Interpolate dpss windows when they are too large to be calculated directl
* `64 <https://github.com/nipy/nitime/issues/64>`_: Pass fir window as a kwarg.
* `39 <https://github.com/nipy/nitime/issues/39>`_: Fix xcorr plot
* `54 <https://github.com/nipy/nitime/issues/54>`_: Reorganize analysis
* `49 <https://github.com/nipy/nitime/issues/49>`_: added basic arithetics to timeseries objects
* `52 <https://github.com/nipy/nitime/issues/52>`_: Reorganization
* `48 <https://github.com/nipy/nitime/issues/48>`_: Fix filter analyzer
* `47 <https://github.com/nipy/nitime/issues/47>`_: Correlation analyzer
* `43 <https://github.com/nipy/nitime/issues/43>`_: Filtfilt
* `42 <https://github.com/nipy/nitime/issues/42>`_: (Not) Biopac
* `41 <https://github.com/nipy/nitime/issues/41>`_: Fix example bugs
* `40 <https://github.com/nipy/nitime/issues/40>`_: Epochslicing2
* `33 <https://github.com/nipy/nitime/issues/33>`_: Epochslicing
* `38 <https://github.com/nipy/nitime/issues/38>`_: Fix snr df bug
* `37 <https://github.com/nipy/nitime/issues/37>`_: Event slicing
* `36 <https://github.com/nipy/nitime/issues/36>`_: Index at bug

Regular issues (10):

* `31 <https://github.com/nipy/nitime/issues/31>`_: tools/make_examples.py runs all the examples every time
* `56 <https://github.com/nipy/nitime/issues/56>`_: Test failure on newer versions of scipy
* `65 <https://github.com/nipy/nitime/issues/65>`_: Prune the nipy/nitime repo from old stragglers
* `57 <https://github.com/nipy/nitime/issues/57>`_: multi_taper_psd with jackknife=True fails with multiple timeseries input
* `59 <https://github.com/nipy/nitime/issues/59>`_: missing parameter docstring in utils.ar_generator
* `58 <https://github.com/nipy/nitime/issues/58>`_: Scale of sigma in algorithms.multi_taper_psd
* `61 <https://github.com/nipy/nitime/issues/61>`_: fail to estimate dpss_windows for long signals
* `34 <https://github.com/nipy/nitime/issues/34>`_: SNR information rates need to be normalized by the frequency resolution
* `45 <https://github.com/nipy/nitime/issues/45>`_: Bugs in CorrelationAnalyzer
* `29 <https://github.com/nipy/nitime/issues/29>`_: Filtering
