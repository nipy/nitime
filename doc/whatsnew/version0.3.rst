
======================================
 Release notes for nitime version 0.3
======================================

Summary of changes
------------------

Version 0.3 of nitime includes several additions and improvements: 

#. Testing: Test coverage of nitime has improved substantially. At this point,
   83 % of the code is executed when the test suite is executed. This includes
   a full 100% execution of all l.o.c. in most of the algorithms
   sub-module. Work led by Ariel Rokem.

#. Style and layout improvements: The layout of the algorithms and analysis
   sub-modules have been simplified and a large majority of the code-base has
   been modified to conform with PEP8 standards. Work done by Ariel Rokem, with
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
   point. Work led by Ariel Rokem, Fernando Perez Killian Koepsell and Paul
   Ivanov.  
   


Contributors to this release
----------------------------

* Alexandre Gramfort
* Ariel Rokem
* Christopher Burns 
* Fernando Perez 
* Michael Waskom 
* Paul Ivanov 
* Yaroslav Halchenko 
* Ariel Rokem
* Jarrod Millman 
* Killian Koepsell
* Mike Trumpis

.. Note::

   This list was generated using::
   
       git log dev/0.3 HEAD --format='* %aN <%aE>' |sed 's/@/\-at\-/' | sed 's/<>//' | sort -u  

   Please let us know if you should appear on this list and do not, so that we
   can add your name in future release notes. 

       
Detailed stats from the github repository
-----------------------------------------

